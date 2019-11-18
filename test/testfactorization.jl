using LinearAlgebra: cholesky
using Kronecker: KroneckerProduct, KroneckerFactorization, FactorizedKroneckerProduct, FactorizedKroneckerPower

@testset "Factorization" begin
    @testset "Basic Properties" begin

        function basic_checks(F::Factorization, K::KronType) where {KronType<:AbstractKroneckerProduct}
            @test typeof(F)<:Factorization # test type hierarchy
            @test typeof(F)<:KroneckerFactorization
            @test issuccess(F)
            @test size(F) == size(K)
            KF = kronecker(F) # reconstruction
            @test KF ≈ K
            @test typeof(KF) == KronType
            @test logdet(F) ≈ logdet(K)
        end

        n = 8
        m = 16
        # 1) testing square kronecker product with square A, B
        A = randn(n, n)
        B = randn(m, m)
        K = A ⊗ B
        F = factorize(K)
        @test typeof(F)<:FactorizedKroneckerProduct
        basic_checks(F, K)

        # 2) testing KroneckerPower
        pow =  3
        K = A ⊗ pow
        F = factorize(K)
        @test typeof(F)<:FactorizedKroneckerPower
        basic_checks(F, K)

    end

    @testset "Solving Linear Systems" begin
        n = 8
        m = 16
        # 1) testing square kronecker product with square A, B
        A = randn(n, n)
        B = randn(m, m)
        K = A ⊗ B
        x = randn(n * m)
        b = K*x
        @test K\b ≈ x

        # rdiv currently does not work because /K relies on calling adjoint of LU
        # @test (x'*K)/K ≈ x'

        # testing least squares solution
        function test_ls_solve(size_A, size_B)
            A = randn(size_A)
            B = randn(size_B)
            x = randn(size(A, 2) * size(B, 2))
            K = kronecker(A, B)
            b = K*x
            b .+= randn(size(b)) # this moves b out of range(K), necessitating least-squares
            xls = K\b
            return K'*(K*xls) ≈ K'b # test via normal equations
        end

        # 2) kronecker product is square, but A, B aren't
        size_A = (m, n)
        size_B = (n, m)
        @test test_ls_solve(size_A, size_B)

        # 3) kronecker product is not square, dimension of x is larger than b
        size_A = (n, n)
        size_B = (n, m)
        @test test_ls_solve(size_A, size_B)

        # 4) kronecker product is not square, dimension of x is smaller than b
        size_A = (m, n)
        size_B = (m, m)
        @test test_ls_solve(size_A, size_B)

        # 5) testing kronecker order > 2
        n = 4
        m = 5
        A = randn(n, n)
        B = randn(m, m)
        C = randn(n, n)
        K = A ⊗ B ⊗ C
        x = randn(n^2 * m)
        b = K*x
        @test K\b ≈ x

        F = factorize(K)
        @test F\b ≈ x # testing factorized solve
        @test F\repeat(b, 1, 2) ≈ repeat(x, 1, 2) # matrix RHS

        # higher order with same sizes not ready
        K = A ⊗ C ⊗ A
        x = randn(n^3)
        b = K*x
        @test K\b ≈ x


        # 6) testing KroneckerPower solve for different powers
        for p = 2:5
            K = kronecker(A, p)
            x = randn(n^p)
            b = K*x
            @test K\b ≈ x

            #7) test ldiv!
            y = similar(b)
            ldiv!(factorize(K), b, y)
            @test x ≈ b
        end


    end

    @testset "Cholesky" begin
        A = [1 0 0.5;
             0 2 0;
             0.5 0 3]

        B = rand(4)
        B *= B'
        B += I

        K = A ⊗ B

        @test isposdef(K)

        KC = cholesky(K)
        @test isposdef(K)
        @test size(K) == (12, 12)

        C = cholesky(collect(K))

        @test det(KC) ≈ det(C)
        @test logdet(KC) ≈ logdet(C)
        @test collect(inv(KC)) ≈ inv(C)
    end
end
