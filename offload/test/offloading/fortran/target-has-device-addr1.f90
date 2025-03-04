!REQUIRES: flang, amdgpu

!Test derived from the sollve test for has-device-addr.

!RUN: %libomptarget-compile-fortran-run-and-check-generic

module m
  use iso_c_binding

contains
  integer function target_has_device_addr()
    integer :: errors
    integer, target :: x
    integer, pointer :: first_scalar_device_addr
    type(c_ptr) :: cptr_scalar1

    integer :: res1, res2

    nullify (first_scalar_device_addr)
    errors = 0
    x = 10

    !$omp target enter data map(to: x)
    !$omp target data use_device_addr(x)
      x = 11
      cptr_scalar1 = c_loc(x)
    !$omp end target data

    call c_f_pointer (cptr_scalar1, first_scalar_device_addr)

    !$omp target map(to: x) map(from: res1, res2) &
    !$omp & has_device_addr(first_scalar_device_addr)
      res1 = first_scalar_device_addr
      res2 = x
    !$omp end target
    target_has_device_addr = errors
    print *, "res1", res1, "res2", res2
  end function
end module


program p
  use m
  integer :: errors

  errors = target_has_device_addr()
  print *, "errors=", errors
end

!CHECK: res1 11 res2 11
!CHECK: errors= 0
