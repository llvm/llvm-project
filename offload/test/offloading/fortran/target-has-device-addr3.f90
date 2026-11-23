!REQUIRES: flang, amdgpu

!RUN: %libomptarget-compile-fortran-run-and-check-generic

function f(x) result(y)
  integer :: x(:)
  integer :: y, z
  x = 0
  y = 11
  !$omp target data map(tofrom: x) use_device_addr(x)
  !$omp target has_device_addr(x) map(tofrom: y)
  y = size(x)
  !$omp end target
  !$omp end target data
end

program main
  interface
    function f(x) result(y)
      integer :: x(:)
      integer :: y
    end function
  end interface
  integer :: x(13)
  integer :: y
  y = f(x)
  print *, "y=", y
end

!CHECK: y= 13
