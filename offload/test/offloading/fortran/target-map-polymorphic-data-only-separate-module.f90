! Offload regression test for data-only mapping of a polymorphic object whose
! type module is compiled separately. The target region only accesses data
! member s%n and does not call type-bound procedures, so the RTTI pass must keep
! type descriptor data available without marking dispatch-table bindings from
! the separate module as declare target.
!
! REQUIRES: flang, amdgpu
!
! RUN: %flang -c -fopenmp -fopenmp-version=61 -fopenmp-targets=%target_triple --offload-arch=native \
! RUN:   %S/../../Inputs/polymorphic-data-only-shapes.f90 -o polymorphic-data-only-shapes.o
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 polymorphic-data-only-shapes.o
! RUN: %libomptarget-run-generic | %fcheck-generic

program main
  use polymorphic_data_only_shapes
  implicit none

  class(shape_t), allocatable :: s

  allocate(circle_t :: s)

  !$omp target map(tofrom: s)
    s%n = s%n + 1
  !$omp end target

  if (s%n /= 1) then
    print *, "======= Test Failed! ======="
    print *, s%n
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(s)
end program main

! CHECK: ======= Test Passed! =======
