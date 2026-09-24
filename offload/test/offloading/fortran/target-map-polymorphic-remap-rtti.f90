! Offload test for remapping a polymorphic descriptor after its dynamic
! type changes.
!
! The test maps a child1_t object, deletes it from the device data
! environment, reallocates the polymorphic variable as child2_t, and maps it
! again.  The target regions use SAME_TYPE_AS to verify that the descriptor
! addendum/RTTI for the second mapping is not stale.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_remap_rtti_mod
  implicit none

  type :: base_t
    integer :: base_value
  end type base_t

  type, extends(base_t) :: child1_t
    integer :: child1_value
  end type child1_t

  type, extends(base_t) :: child2_t
    integer :: child2_value
  end type child2_t

end module polymorphic_remap_rtti_mod

program main
  use polymorphic_remap_rtti_mod
  implicit none

  class(base_t), allocatable :: obj
  type(child1_t) :: child1_mold
  type(child2_t) :: child2_mold
  logical :: first_is_child1
  logical :: first_is_child2
  logical :: second_is_child1
  logical :: second_is_child2

  allocate(child1_t :: obj)
  obj%base_value = 1

  select type (obj)
  type is (child1_t)
    obj%child1_value = 10
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  first_is_child1 = .false.
  first_is_child2 = .true.

  !$omp target enter data map(to: obj)

  !$omp target map(tofrom: obj, first_is_child1, first_is_child2) map(to: child1_mold, child2_mold)
    first_is_child1 = same_type_as(obj, child1_mold)
    first_is_child2 = same_type_as(obj, child2_mold)
  !$omp end target

  !$omp target exit data map(delete: obj)

  if (.not. first_is_child1) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (first_is_child2) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  deallocate(obj)
  allocate(child2_t :: obj)
  obj%base_value = 2

  select type (obj)
  type is (child2_t)
    obj%child2_value = 20
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  second_is_child1 = .true.
  second_is_child2 = .false.

  !$omp target enter data map(to: obj)

  !$omp target map(tofrom: obj, second_is_child1, second_is_child2) map(to: child1_mold, child2_mold)
    second_is_child1 = same_type_as(obj, child1_mold)
    second_is_child2 = same_type_as(obj, child2_mold)
  !$omp end target

  !$omp target exit data map(delete: obj)

  if (second_is_child1) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (.not. second_is_child2) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
