! Validate that a device pointer obtained via omp_get_mapped_ptr can be used
! inside a TARGET region with the is_device_ptr clause.
! REQUIRES: flang, amdgcn-amd-amdhsa

! RUN: %libomptarget-compile-fortran-run-and-check-generic

module mod
  implicit none
  integer, parameter :: n = 4
contains
  subroutine kernel(dptr)
    use iso_c_binding, only : c_ptr, c_f_pointer
    implicit none

    type(c_ptr) :: dptr
    integer, dimension(:), pointer :: b
    integer :: i

    b => null()

    !$omp target is_device_ptr(dptr)
      call c_f_pointer(dptr, b, [n])
      do i = 1, n
        b(i) = b(i) + 1
      end do
    !$omp end target
  end subroutine kernel
end module mod

program is_device_ptr_target
  use iso_c_binding, only : c_ptr, c_loc, c_f_pointer
  use omp_lib, only: omp_get_default_device, omp_get_mapped_ptr
  use mod, only: kernel, n
  implicit none

  integer, dimension(n), target :: a
  integer :: dev
  type(c_ptr) :: dptr

  a = [2, 4, 6, 8]
  print '("BEFORE:", I3)', a

  dev = omp_get_default_device()

  !$omp target data map(tofrom: a)
    dptr = omp_get_mapped_ptr(c_loc(a), dev)
    call kernel(dptr)
  !$omp end target data

  print '("AFTER: ", I3)', a

  if (all(a == [3, 5, 7, 9])) then
    print '("PASS")'
  else
    print '("FAIL   ", I3)', a
  end if

end program is_device_ptr_target

!CHECK: PASS
