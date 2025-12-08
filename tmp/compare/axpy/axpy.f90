program axpy_test
  implicit none

  integer, parameter :: N = 10000
  real(4), parameter :: a = 2.0
  real(4), allocatable :: x(:), y(:), z(:)
  integer :: i

  allocate(x(N), y(N), z(N))
  do i = 1, N
    x(i) = real(i, kind=4)
    y(i) = 0.0
    z(i) = 0.0
  end do

  call vector_axpy(N, a, x, y, z)

  ! Print some results to verify
  print *, "First 5 elements of result:"
  do i = 1, 5
    print *, "z(", i, ") = ", z(i)
  end do

  deallocate(x, y, z)

contains

  subroutine vector_axpy(n, a, x, y, z)
    integer, intent(in) :: n
    real(4), intent(in) :: a
    real(4), intent(in) :: x(n)
    real(4), intent(in) :: y(n)
    real(4), intent(out) :: z(n)

    !$omp target teams coexecute
    z = a * x + y
    !$omp end target teams coexecute 
  end subroutine
end program axpy_test
