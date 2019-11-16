#=
Created on Friday 26 July 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Standard matrix factorization algorithms applied on Kronecker systems.
=#

import LinearAlgebra: Factorization, factorize

# KroneckerFactorization should not be a subtype of AbstractKroneckerProduct,
# just as Factorization is not a subtype of AbstractMatrix
# KroneckerFactorization is abstract, keeping analogy with Factorization in LinearAlgebra
abstract type KroneckerFactorization{T} <: Factorization{T} end

# FactorizedKronecker is a concrete type holding arbitrary factorizations
struct FactorizedKronecker{T, TA<:Factorization{T}, TB<:Factorization{T}} <: KroneckerFactorization{T}
    A::TA
    B::TB
end

KroneckerProduct(F::Factorization) = AbstractMatrix(F) # base case of recursion
KroneckerProduct(K::FactorizedKronecker) = KroneckerProduct(KroneckerProduct(K.A), KroneckerProduct(K.B)) # cast to KroneckerProduct
kronecker(K::FactorizedKronecker) = KroneckerProduct(K)

#TODO:
# allow log, logdet, getmatrices, on const KronProdOrFact = Union{AbstractKroneckerProduct, KroneckerFactorization}

import LinearAlgebra: issuccess, Matrix
issuccess(K::FactorizedKronecker) = issuccess(K.A) && issuccess(K.B)
collect(K::FactorizedKronecker) = collect(kronecker(K))
Matrix(K::FactorizedKronecker) = collect(K)

factorize(K::AbstractKroneckerProduct) = FactorizedKronecker(factorize(K.A), factorize(K.B))

# cholesky
function cholesky(K::AbstractKroneckerProduct, ::Val{false}=Val(false); check::Bool = true)
    f(A) = cholesky(A, Val(false), check = check)
    FactorizedKronecker(f(K.A), f(K.B))
end

# pivoted cholesky
function cholesky(K::AbstractKroneckerProduct, ::Val{true}; tol = 0.0, check::Bool = true)
    f(A) = cholesky(A, Val(true), tol = tol, check = check)
    FactorizedKronecker(f(K.A), f(K.B))
end

function qr(K::AbstractKroneckerProduct, v::V = Val(false)) where {V<:Union{Val{true}, Val{false}}}
    f(A) = qr(A, v)
    FactorizedKronecker(f(K.A), f(K.B))
end

# in place versions
function cholesky!(K::AbstractKroneckerProduct, ::Val{false}=Val(false); check::Bool = true)
    f(A) = cholesky!(A, Val(false), check = check)
    FactorizedKronecker(f(K.A), f(K.B))
end

# pivoted cholesky
function cholesky!(K::AbstractKroneckerProduct, ::Val{true}; tol = 0.0, check::Bool = true)
    f(A) = cholesky!(A, Val(true), tol = tol, check = check)
    FactorizedKronecker(f(K.A), f(K.B))
end

# overwrites input Kronecker product
function qr!(K::AbstractKroneckerProduct, v::V = Val(false)) where {V<:Union{Val{true}, Val{false}}}
    f(A) = qr!(A, v)
    FactorizedKronecker(f(K.A), f(K.B))
end

# TODO: extend methods for all LinearAlgebra factorizations: svd, bunchkaufman, lu,

# need to implement ldiv!, rdiv!, for \, / to work for factorizations of Kronecker matrices
import LinearAlgebra:ldiv! #, rdiv!, \, /
# do not need to define this because LinearAlgebra takes care of it
# function ldiv!(y::AbstractVector, K::KroneckerFactorization, c::AbstractVector)
#     copyto!(y, c)
#     ldiv!(K, y)
# end

# overwrites B with solution to Ax = c
# cases:
# 1) everything is square
# 2) x is smaller than c
# 3) c is smaller than x
# 4) c is smaller than x, but it has enough memory to hold x, we should assume this
# we assume that c has enough memory to hold x,
function ldiv!(A::KroneckerFactorization, c::AbstractVector)
    # if issquare(K.A) && issquare(K.B)
    #     n = size(K.A, 1)
    #     m = size(K.B, 1)
    #     C = reshape(c, (s, n))
    #     ldiv!(K.B, C) # overwrites C
    #     rdiv!(C, K.A') # overwrites C
    # else
    n, m = size(K.A)
    s, t = size(K.B)
    # C = reshape(c, (s, :))
    C1 = reshape(view(c, 1:s*n), (s,n))
    C2 = reshape(view(c, 1:t*n), (t,n)) # could instead have (s,m) matrix here have to decide what is more efficient
    C3 = reshape(view(c, 1:t*m), (t,m))
    ldiv!(C2, K.B, C1) # overwrites C
    rdiv!(C3, K.A') # overwrites C , not all factorizations define rdiv!
    # end
end
    # non-square case is complicated, easiest if n > m, s > t -> can store in C
    # n, m = size(K.A)
    # s, t = size(K.B)

    # elseif n ≤ m
    #     C = reshape(c, (s, n)) # reshaping does not allocate
    #     # vec((K.B \ C) / K.A')
    #     T = (K.B \ C) # is of size (t, n) # (size(K.B, 2), size(C, 2))
    #
    #     # X = T / K.A' is of size (size(K.T, 1), size(K.A, 2)) = (t, m)
    #     # Y must be of size (t, m) # (size(K.B, 2), size(K.A, 2))
    #     ldiv!(K.B, C)
    #
    #     Y = reshape(y, (t, :))
    #     Y_view = @view Y[1:t, 1:n]
    #     copyto!(Y_view, )
    #
    # else
    #     T = C / K.A' # is of size (s, m)

    # end
    # require_one_based_indexing(Y, B)
    # m, n = size(A, 1), size(A, 2)
    # if m > n
    #     Bc = copy(B)
    #     ldiv!(A, Bc)
    #     return copyto!(Y, view(Bc, 1:n, :))
    # else
    #     return ldiv!(A, copyto!(Y, view(B, 1:m, :)))
    # end


function LinearAlgebra.:\(K::AbstractKroneckerProduct{T}, c::AbstractVector{T}) where {T}
    size(K, 2) != length(c) && throw(DimensionMismatch("size(K, 2) != length(c)"))
    C = reshape(c, size(K.B, 1), size(K.A, 1)) # matricify
    return vec((K.B \ C) / K.A') #(A ⊗ B)vec(X) = vec(C) <=> BXA' = C => X = B^{-1} C A'^{-1}
end
# SOLVING
# Kx = c
function LinearAlgebra.:\(K::AbstractKroneckerProduct{T}, c::AbstractVector{T}) where {T}
    size(K, 1) != length(c) && throw(DimensionMismatch("size(K, 1) != length(c)"))
    C = reshape(c, size(K.B, 1), size(K.A, 1)) # matricify
    return vec((K.B \ C) / K.A') #(A ⊗ B)vec(X) = vec(C) <=> BXA' = C => X = B^{-1} C A'^{-1}
end

import LinearAlgebra: log, logdet

function det(K::FactorizedKronecker{T}) where {T}
    if issquare(K.A) && issquare(K.B)
        m = size(K.A)[1]
        n = size(K.B)[1]
        return det(K.A)^n * det(K.B)^m
    else
        return zero(T)
    end
end

"""
    logdet(K::FactorizedKronecker)

Compute the logarithm of the determinant of a Kronecker product.
"""
function logdet(K::FactorizedKronecker{T}) where {T}
    if issquare(K.A) && issquare(K.B)
        m = size(K.A)[1]
        n = size(K.B)[1]
        return n * logdet(K.A) + m * logdet(K.B)
    else
        return real(T)(-Inf)
    end
end

# these need to change in base
issquare(K::AbstractKroneckerProduct) = size(K, 1) == size(K, 2)

function LinearAlgebra.tr(K::AbstractKroneckerProduct)
    !issquare(K) && throw(
                DimensionMismatch(
                    "kronecker system is not square: dimensions are" * size(K)))
    if issquare(K.A) && issquare(K.B)
        return tr(K.A) * tr(K.B)
    else
        return sum(diag(K)) # fallback
    end
end


# now we can define specific factorizations as subtypes
# struct CholeskyKronecker{T} <: Factorization{T}
#     A::Union{Cholesky{T}, CholeskyKronecker{T}}
#     B::Union{Cholesky{T}, CholeskyKronecker{T}}
# end

# CHOLESKY DECOMPOSITION
# ----------------------


# import LinearAlgebra: Cholesky, cholesky
#
# struct CholeskyKronecker{T<:Union{Cholesky,FactorizedKronecker},S<:Union{Cholesky,FactorizedKronecker}} <: FactorizedKronecker{Float64}
#     A::T
#     B::S
# end
#
# issquare(C::Cholesky) = true
#
# function Base.getproperty(C::CholeskyKronecker, d::Symbol)
#     if d in [:U, :L, :UL]
#         return kronecker(getproperty(C.A, d), getproperty(C.B, d))
#     elseif d in [:A, :B]
#         return getfield(C, d)
#     else
#         throw(ArgumentError("Attribute :$d not supported (only :A, :B, :UL, :U or :L)"))
#     end
# end
#
# function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, C::CholeskyKronecker)
#     summary(io, C); println(io)
#     println(io, "U factor:")
#     show(io, mime, getproperty(C, :U))
# end
#
# """
#     cholesky(K::AbstractKroneckerProduct; check = true)
#
# Wrapper around `cholesky` from the `LinearAlgebra` package. Performs Cholesky
# on the matrices of a `AbstractKroneckerProduct` instances and returns a
# `CholeskyKronecker` type. Similar to `Cholesky`, `size`, `\\`, `inv`, `det`,
# and `logdet` are overloaded to efficiently work with this type.
# """
# function cholesky(K::AbstractKroneckerProduct; check = true)
#     squarecheck(K)
#     A, B = getmatrices(K)
#     return CholeskyKronecker(cholesky(A, check=check),
#                             cholesky(B, check=check))
# end
