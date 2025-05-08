!REQUIRES: flang, amdgpu

!RUN: %libomptarget-compile-fortran-run-and-check-generic

program m
  complex(kind=8) :: x
  x = (1.0, 2.0)
!$omp target
  x = (-1.0, -2.0)
!$omp end target
  print *, "x=", x
end program

! The host variable "x" should be passed to the kernel as "firstprivate",
! hence the kernel should have its own copy of it. This is in contrast to
! other cases where implicitly mapped variables have the TOFROM map-type.

! Make sure that the target region didn't overwrite the host variable.

!CHECK: x= (1.,2.)
