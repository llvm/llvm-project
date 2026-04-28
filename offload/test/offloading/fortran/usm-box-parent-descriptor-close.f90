! Test for PR fixing close flag on descriptor members for box parents in USM
! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-generic -fopenmp-force-usm
! RUN: env LIBOMPTARGET_INFO=16 HSA_XNACK=1 %libomptarget-run-generic 2>&1 | %fcheck-generic

module m
    implicit none
    integer :: ng
    type :: gt
        integer :: k
    end type
    type(gt), allocatable :: g(:)
    !$omp declare target(ng, g)
    type :: f
        real, allocatable :: a(:)
    end type
end module m

program r
    use m
    implicit none
    integer :: i
    type(f), target :: u(2)
    integer :: ig
    real, contiguous, pointer :: p(:)

    ng = 1
    allocate(g(1))
    g(1)%k = 1

    do i = 1, 2
        allocate(u(i)%a(1), source=0.0)
    end do
    u(1)%a(1) = 1.0
    u(2)%a(1) = -1.0

    !$omp target enter data map(to: g, ng, u(1)%a, u(2)%a)

    !$omp target teams distribute private(ig, p)
    do ig = 1, ng
        p(1:1) => u(2)%a(1:1)
        p(1) = 3.14
    end do
    !$omp end target teams distribute

    ! CHECK: PluginInterface device {{[0-9]+}} info: Launching kernel
    ! CHECK: Result: 3.14
    print *, "Result: ", u(2)%a(1)
end program r
