! Offloading test which aims to test that an allocatable/descriptor type map
! will allow the appropriate slicing behaviour.
! REQUIRES: flang, amdgpu

subroutine slice_writer(n, a, b, c)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: a(n)
    real(8), intent(in) :: b(n)
    real(8), intent(out) :: c(n)
    integer :: i

    !$omp target teams distribute parallel do
    do i=1,n
       c(i) = b(i) + a(i)
    end do
end subroutine slice_writer

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none
    real(kind=8), allocatable :: a(:,:,:)
    integer :: i, j, k, idx, idx1, idx2, idx3

    i=50
    j=100
    k=2

    allocate(a(1:i,1:j,1:k))

    do idx1=1, i
        do idx2=1, j
            do idx3=1, k
                a(idx1,idx2,idx3) = idx2
            end do
        end do
    end do

    do idx=1,k
        !$omp target enter data map(alloc: a(1:i,:, idx))

        !$omp target update to(a(1:i, 1:30, idx), &
        !$omp&                 a(1:i, 61:100, idx))

        call slice_writer(i, a(:, 1, idx), a(:, 61, idx), a(:, 31, idx))
        call slice_writer(i, a(:, 30, idx), a(:, 100, idx), a(:, 60, idx))

        !$omp target update from(a(1:i, 31:60, idx))
        !$omp target exit data map(delete: a(1:i, :, idx))

        print *, a(1, 31, idx), a(2, 31, idx), a(i, 31, idx)
        print *, a(1, 60, idx), a(2, 60, idx), a(i, 60, idx)
    enddo

    deallocate(a)
end program

! CHECK: 62. 62. 62.
! CHECK: 130. 130. 130.
! CHECK: 62. 62. 62.
! CHECK: 130. 130. 130.
