! Offload test for polymorphic Fortran descriptor RTTI without SELECT TYPE.
!
! This uses EXTENDS_TYPE_OF on device to verify that the descriptor addendum's
! derived_type pointer is attached to the canonical device TypeDescriptor
! global.  The test avoids SELECT TYPE in the target regions.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_rtti_intrinsics_mod
  implicit none

  type :: base_t
    integer :: base_value
  end type base_t

  type, extends(base_t) :: child_t
    integer :: child_value
  end type child_t

contains

  subroutine check_dummy_addendum(obj, extends_base, base_extends_obj)
    class(base_t), intent(inout) :: obj
    logical, intent(out) :: extends_base
    logical, intent(out) :: base_extends_obj
    type(base_t) :: base_mold

    extends_base = .false.
    base_extends_obj = .true.

    !$omp target map(tofrom: obj, extends_base, base_extends_obj) map(to: base_mold)
      extends_base = extends_type_of(obj, base_mold)
      base_extends_obj = extends_type_of(base_mold, obj)
      obj%base_value = obj%base_value + 1
    !$omp end target
  end subroutine check_dummy_addendum

end module polymorphic_rtti_intrinsics_mod

program main
  use polymorphic_rtti_intrinsics_mod
  implicit none

  class(base_t), allocatable :: obj
  type(base_t) :: base_mold
  type(child_t) :: child_actual
  logical :: extends_base
  logical :: base_extends_child
  logical :: dummy_extends_base
  logical :: dummy_base_extends_obj

  allocate(child_t :: obj)
  obj%base_value = 10

  extends_base = .false.
  base_extends_child = .true.

  !$omp target enter data map(to: obj)

  !$omp target map(tofrom: obj, extends_base, base_extends_child) map(to: base_mold)
    extends_base = extends_type_of(obj, base_mold)
    base_extends_child = extends_type_of(base_mold, obj)
    obj%base_value = obj%base_value + 1
  !$omp end target

  !$omp target exit data map(from: obj)

  if (.not. extends_base) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (base_extends_child) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (obj%base_value /= 11) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  child_actual%base_value = 20
  child_actual%child_value = 30
  call check_dummy_addendum(child_actual, dummy_extends_base, dummy_base_extends_obj)

  if (.not. dummy_extends_base) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (dummy_base_extends_obj) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (child_actual%base_value /= 21) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (child_actual%child_value /= 30) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
