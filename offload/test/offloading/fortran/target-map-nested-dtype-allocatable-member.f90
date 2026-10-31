! Offloading test checking that mapping two variables of a derived type
! containing a nested derived type with an allocatable member does not crash
! during MapInfoFinalization. The member maps for the inner derived type are
! not direct target operands; the pass must skip them when walking record types
! with allocatable members.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: inner_type
      integer, allocatable :: a
    end type inner_type

    type :: outer_type
      type(inner_type) :: b
    end type outer_type

    type(outer_type) :: var1, var2

    allocate(var1%b%a)
    allocate(var2%b%a)
    var1%b%a = 10
    var2%b%a = 20

    !$omp target map(tofrom: var1%b, var2%b)
      var1%b%a = var1%b%a + var2%b%a
    !$omp end target

    print *, var1%b%a

end program main

!CHECK: 30
