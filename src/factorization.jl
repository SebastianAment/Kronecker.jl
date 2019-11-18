import LinearAlgebra: Factorization, factorize
import LinearAlgebra: issuccess, Matrix, size

# KroneckerFactorization should not be a subtype of AbstractKroneckerProduct,
# just as Factorization is not a subtype of AbstractMatrix
# KroneckerFactorization is abstract, keeping analogy with Factorization in LinearAlgebra
abstract type KroneckerFactorization{T} <: Factorization{T} end

# export KroneckerFactorization, FactorizedKroneckerProduct, FactorizedKroneckerPower

# FactorizedKroneckerProductProduct is a concrete type holding arbitrary factorizations
# FactOrAdj = Union{Factorization, Adjoint{}}
struct FactorizedKroneckerProduct{T,
        TA<:Union{Factorization{T}, Adjoint{T, <:Factorization}},
        TB<:Union{Factorization{T}, Adjoint{T, <:Factorization}}} <: KroneckerFactorization{T}
    A::TA
    B::TB
end

struct FactorizedKroneckerPower{T,
        TA<:Union{Factorization{T}, Adjoint{T, <:Factorization}},
        N} <: KroneckerFactorization{T}
    A::TA
    pow::Int
    function FactorizedKroneckerPower(A, pow::Integer) where {T}
       @assert pow ≥ 2 "FactorizedKroneckerPower only makes sense for powers greater than 1"
       return new{eltype(A), typeof(A), pow}(A, pow)
     end
end

# this is stylistically not the best, but let's keep it for now
getmatrices(K::FactorizedKroneckerProduct) = (K.A, K.B)
getmatrices(K::FactorizedKroneckerPower) = (K.A, FactorizedKroneckerPower(K.A, K.pow-1))
getmatrices(K::FactorizedKroneckerPower{T, TA, 2}) where {T, TA} = (K.A, K.A)

# For KroneckerSum, we can use a function in LinearAlgebra to solve AX + BX = C
# struct FactorizedKroneckerSum{T<:Any, TA<:Factorization{T}, TB<:Factorization{T}} <: KroneckerFactorization{T}
#     A::TA
#     B::TB
# end

# generic
kronecker(K::KroneckerFactorization) = KroneckerProduct(K)
KroneckerProduct(F::Factorization) = AbstractMatrix(F) # base case of recursion
collect(K::KroneckerFactorization) = collect(kronecker(K))
Matrix(K::KroneckerFactorization) = collect(K)
size(K::KroneckerFactorization) = (size(K, 1), size(K, 2))

issquare(A::Factorization) = size(A, 1) == size(A, 2)
# checks if all component matrices of a KroneckerProduct have the same size
samesize(K::Union{AbstractMatrix, Factorization}) = (true, size(K))
function samesize(K::Union{AbstractKroneckerProduct, KroneckerFactorization})
    A, B = getmatrices(K)
    ba, sa = samesize(A)
    bb, sb = samesize(B)
    return (ba && bb) && (sa == sb), sa
end

# FactorizedKroneckerProduct
# factorize(K::AbstractKroneckerProduct) = (A, B = getmatrices(K); FactorizedKroneckerProduct(factorize(A), factorize(B)))
factorize(K::AbstractKroneckerProduct) = FactorizedKroneckerProduct(factorize(K.A), factorize(K.B))
KroneckerProduct(K::FactorizedKroneckerProduct) = KroneckerProduct(KroneckerProduct(K.A), KroneckerProduct(K.B)) # cast to KroneckerProduct
issuccess(K::FactorizedKroneckerProduct) = issuccess(K.A) && issuccess(K.B)
size(K::FactorizedKroneckerProduct, d::Int) = size(K.A, d) * size(K.B, d)
adjoint(K::FactorizedKroneckerProduct) = FactorizedKroneckerProduct(adjoint(K.A), adjoint(K.B))

# FactorizedKroneckerPower
factorize(K::KroneckerPower) = FactorizedKroneckerPower(factorize(K.A), K.pow)
KroneckerProduct(K::FactorizedKroneckerPower) = KroneckerPower(KroneckerProduct(K.A), K.pow) # cast to KroneckerPower
issuccess(K::FactorizedKroneckerPower) = issuccess(K.A)
size(K::FactorizedKroneckerPower, d::Int) = size(K.A, 1)^K.pow
adjoint(K::FactorizedKroneckerPower) = FactorizedKroneckerProduct(adjoint(K.A), K.pow)

#TODO:
const KronProdOrFact = Union{KroneckerProduct, FactorizedKroneckerProduct}
const KronPowOrFact = Union{KroneckerPower, FactorizedKroneckerPower}
const AbstractKronOrFact = Union{AbstractKroneckerProduct, KroneckerFactorization}

import LinearAlgebra: cholesky, cholesky!, qr, qr!#, svd, eigen, bunchkaufman, lu,
# TODO: 1) extend methods for all LinearAlgebra factorizations: svd, eigen,
# 2) extend them for KroneckerPower, metaprogramming?
# 3) need to test for complex valued kronecker matrices, might need to replace
# some adjoints with transposes in the code.

# cholesky
function cholesky(K::AbstractKroneckerProduct, ::Val{false}=Val(false); check::Bool = true)
    f(A) = cholesky(A, Val(false), check = check)
    FactorizedKroneckerProduct(f(K.A), f(K.B))
end

# in place cholesky
function cholesky!(K::AbstractKroneckerProduct, ::Val{false}=Val(false); check::Bool = true)
    f(A) = cholesky!(A, Val(false), check = check)
    FactorizedKroneckerProduct(f(K.A), f(K.B))
end

# pivoted cholesky
function cholesky(K::AbstractKroneckerProduct, ::Val{true}; tol = 0.0, check::Bool = true)
    f(A) = cholesky(A, Val(true), tol = tol, check = check)
    FactorizedKroneckerProduct(f(K.A), f(K.B))
end

# in-place pivoted cholesky
function cholesky!(K::AbstractKroneckerProduct, ::Val{true}; tol = 0.0, check::Bool = true)
    f(A) = cholesky!(A, Val(true), tol = tol, check = check)
    FactorizedKroneckerProduct(f(K.A), f(K.B))
end

# qr
function qr(K::AbstractKroneckerProduct, v::V = Val(false)) where {V<:Union{Val{true}, Val{false}}}
    f(A) = qr(A, v)
    FactorizedKroneckerProduct(f(K.A), f(K.B))
end

# in-place qr
function qr!(K::AbstractKroneckerProduct, v::V = Val(false)) where {V<:Union{Val{true}, Val{false}}}
    f(A) = qr!(A, v)
    FactorizedKroneckerProduct(f(K.A), f(K.B))
end


# need to implement ldiv!, rdiv!, for \, / to work for factorizations of Kronecker matrices
import LinearAlgebra: \, ldiv!
# do not need to define this because LinearAlgebra takes care of it
# function ldiv!(y::AbstractVector, K::KroneckerFactorization, c::AbstractVector)
#     copyto!(y, c)
#     ldiv!(K, y)
# end
#
\(K::AbstractKroneckerProduct, C::Union{AbstractVector, AbstractMatrix}) = factorize(K) \ C

function \(K::KroneckerFactorization, c::AbstractVector)
    size(K, 1) != length(c) && throw(DimensionMismatch("size(K, 1) != length(c)"))
    if typeof(K) <: FactorizedKroneckerPower #samesize(K)[1]
        x = copy(c)
        y = similar(x) # pre-allocate only temporary
        ldiv!(K, x, y)
    else # allocating fallback
        A, B = getmatrices(K)
        C = reshape(c, (size(B, 1), size(A, 1)))
        x = vec((A \ (B \ C)')')
    end
    return x
end

# extending to matrix case
function \(K::KroneckerFactorization, C::AbstractMatrix)
    n = size(C, 2)
    X = zeros(eltype(K), (size(K, 2), n))
    for i = 1:n
        X[:,i] = K \ C[:,i]
    end
    return X
end

# ldiv!(K, x, y) computes the vector K^{-1}x and stores it in x
# ldiv! is only defined if all component matrices of K are square
# in that case, y is the only temporary required, and ldiv! is non-allocating
function ldiv!(K::FactorizedKroneckerPower, x::AbstractVector, y::AbstractVector)
    !samesize(K)[1] && throw(DimensionMismatch("ldiv! requires all component matrices to have the same size"))
    n = size(K.A, 1)
    X = reshape(x, (n, :)) # matricify, does not allocate
    Y = reshape(y, (n, :))
    ldiv!(K, X, Y)
end

# this signature is problematic, because we can't defined a matrix ldiv ...
# we can either get rid of this or make an "invisible helper"
function ldiv!(K::FactorizedKroneckerPower, X::AbstractMatrix, Y::AbstractMatrix)
    n = size(K.A, 1)
    for i = 1:K.pow
        ldiv!(K.A, X) # have to check that K.A is a Matrix factorization, not itself KroneckerFactorization
        Y .= reshape(X', (n, :)) # broadcast assignment here is critical
        copyto!(X, Y)
    end
    return
end

function ldiv!(K::FactorizedKroneckerProduct, x::AbstractVector, y::AbstractVector)
    !samesize(K)[1] && throw(DimensionMismatch("ldiv! requires all component matrices to have the same size"))
    A, B = getmatrices(K)
    n = size(A, 1)
    m = size(B, 1)
    X = reshape(x, (m, n))
    Y = reshape(y, (m, n))
    ldiv!(K, X, Y)
end

# Y is the temporary we need, of the same size as X
# might in addition need to restrict to square
function ldiv!(K::FactorizedKroneckerProduct, X::AbstractMatrix, Y::AbstractMatrix)
    A, B = getmatrices(K)
    println("here")
    n = checksquare(A)
    if typeof(B) <: FactorizedKroneckerProduct
        ldiv!(B, X, Y) # recursive Kronecker solve
    else
        ldiv!(B, X) # base case where B is a regular matrix factorization
    end
    Y .= reshape(X', (n, :)) # broadcast assignment here is critical
    copyto!(X, Y)

    if typeof(A) <: FactorizedKroneckerProduct
        ldiv!(A, X, Y)
    else
        ldiv!(A, X)
    end
    Y .= reshape(X', (n, :))
    copyto!(X, Y)
end

# need base case for matrices
# function ldiv!(K::Factorization, C::AbstractMatrix, Y::AbstractMatrix)
#     !samesize(K) && throw(DimensionMismatch("ldiv! requires all component matrices to be square")
#     A, B = getmatrices(K)
#     ldiv!(B, X, Y) # overwrites C
#     Y .= reshape(X', (n, :)) # broadcast assignment here is critical
#     copyto!(X, Y)
#     ldiv!(A, X, Y) # overwrites C, need to reshape C
#     Y .= reshape(X', (n, :)) # broadcast assignment here is critical
#     copyto!(X, Y)
# end

# else
    # n, m = size(K.A)
    # s, t = size(K.B)
    # # C = reshape(c, (s, :))
    # C1 = reshape(view(c, 1:s*n), (s,n))
    # C2 = reshape(view(c, 1:t*n), (t,n)) # could instead have (s,m) matrix here have to decide what is more efficient
    # C3 = reshape(view(c, 1:t*m), (t,m))
    # ldiv!(C2, K.B, C1) # overwrites C
    # ldiv!(K.A, C3) # overwrites C , not all factorizations define rdiv!
# y .= vec((A \ (B \ C)')') # allocating fallback
# end

# now we can define specific factorizations as subtypes
# struct CholeskyKronecker{T} <: Factorization{T}
#     A::Union{Cholesky{T}, CholeskyKronecker{T}}
#     B::Union{Cholesky{T}, CholeskyKronecker{T}}
# end

# redefinition of things in base for factorizations
function LinearAlgebra.isposdef(K::Union{AbstractKroneckerProduct, KroneckerFactorization})
    A, B = getmatrices(K)
    return isposdef(A) && isposdef(B)
end

function LinearAlgebra.det(K::KroneckerFactorization{T}) where {T}
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        m = size(A, 1)
        n = size(B, 1)
        return det(A)^n * det(B)^m
    else
        return zero(T)
    end
end

function LinearAlgebra.logdet(K::KroneckerFactorization{T}) where {T}
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        m = size(A, 1)
        n = size(B, 1)
        return n * logdet(A) + m * logdet(B)
    else
        return real(T)(-Inf)
    end
end

function inv(K::Union{AbstractKroneckerProduct, KroneckerFactorization})
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        return KroneckerProduct(inv(A), inv(B))
    else
        throw(SingularException(1))
    end
end

function LinearAlgebra.det(K::Union{KroneckerPower, FactorizedKroneckerPower})
    n = checksquare(K.A)
    A, pow = K.A, K.pow
    p = pow * n^(pow-1)
    return det(K.A)^p
end

function LinearAlgebra.logdet(K::Union{KroneckerPower, FactorizedKroneckerPower})
    n = checksquare(K.A)
    A, pow = K.A, K.pow
    p = pow * n^(pow-1)
    return p * logdet(K.A)
end
