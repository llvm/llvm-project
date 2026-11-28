! Offload regression test for a contiguous ranged map of a multidimensional
! polymorphic array. The mapped section is contiguous in Fortran column-major
! storage because only the first dimension varies.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_contiguous_range_mod
  implicit none

  type :: base_t
    integer :: x = 0
  end type base_t
end module target_map_polymorphic_contiguous_range_mod

program main
  use target_map_polymorphic_contiguous_range_mod
  implicit none

  class(base_t), allocatable :: obj(:,:)
  integer :: i, j
  logical :: failed

  allocate(base_t :: obj(5,5))

  do j = 1, 5
    do i = 1, 5
      obj(i,j)%x = 100 * i + j
    end do
  end do

  !$omp target map(tofrom: obj(3:4,2))
    obj(3,2)%x = 320
    obj(4,2)%x = 420
  !$omp end target

  failed = .false.
  if (obj(3,2)%x /= 320) failed = .true.
  if (obj(4,2)%x /= 420) failed = .true.

  if (obj(2,2)%x /= 202) failed = .true.
  if (obj(5,2)%x /= 502) failed = .true.
  if (obj(3,3)%x /= 303) failed = .true.

  if (failed) then
    print *, "======= Test Failed! ======="
    print *, obj(2,2)%x, obj(3,2)%x, obj(4,2)%x, obj(5,2)%x, obj(3,3)%x
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(obj)
end program main

! CHECK: ======= Test Passed! =======
