! Validate that a device pointer obtained via omp_get_mapped_ptr can be used
! inside a TARGET region with the is_device_ptr clause.
! REQUIRES: flang, amdgcn-amd-amdhsa

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program is_device_ptr_target
  use iso_c_binding, only : c_ptr, c_loc
  implicit none

  interface
    function omp_get_mapped_ptr(host_ptr, device_num)                       &
        bind(C, name="omp_get_mapped_ptr")
      use iso_c_binding, only : c_ptr, c_int
      type(c_ptr) :: omp_get_mapped_ptr
      type(c_ptr), value :: host_ptr
      integer(c_int), value :: device_num
    end function omp_get_mapped_ptr
  end interface

  integer, parameter :: n = 4
  integer, parameter :: dev = 0
  integer, target :: a(n)
  type(c_ptr) :: dptr
  integer :: flag

  a = [2, 4, 6, 8]
  flag = 0

  !$omp target data map(tofrom: a, flag)
    dptr = omp_get_mapped_ptr(c_loc(a), dev)

    !$omp target is_device_ptr(dptr) map(tofrom: flag)
      flag = flag + 1
    !$omp end target
  !$omp end target data

  if (flag .eq. 1 .and. all(a == [2, 4, 6, 8])) then
    print *, "PASS"
  else
    print *, "FAIL", a
  end if

end program is_device_ptr_target

!CHECK: PASS
