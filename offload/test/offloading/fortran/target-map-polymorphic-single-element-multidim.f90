! Offload regression test for mapping a single element of a multidimensional
! polymorphic allocatable array. The runtime-sized polymorphic map is collapsed
! to a byte map, but the offset must be reconstructed from the descriptor's
! original per-dimension byte strides, not from the selected section extents
! which are all one for scalar subscripts.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_single_element_multidim_mod
  implicit none

  type :: base_t
    integer :: x = 0
  end type base_t
end module target_map_polymorphic_single_element_multidim_mod

program main
  use target_map_polymorphic_single_element_multidim_mod
  implicit none

  class(base_t), allocatable :: obj(:,:)
  logical :: failed

  allocate(base_t :: obj(5,5))
  obj(:,:)%x = 0

  !$omp target map(tofrom: obj(3,2))
    obj(3,2)%x = 32
  !$omp end target

  failed = .false.
  if (obj(3,2)%x /= 32) failed = .true.
  if (obj(4,1)%x /= 0) failed = .true.

  if (failed) then
    print *, "======= Test Failed! ======="
    print *, obj(3,2)%x, obj(4,1)%x
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(obj)
end program main

! CHECK: ======= Test Passed! =======
