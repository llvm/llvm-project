! Offload test for polymorphic Fortran pointer descriptor RTTI without
! SELECT TYPE in the target region.
!
! This uses EXTENDS_TYPE_OF on device to verify that a polymorphic pointer
! descriptor's addendum derived_type pointer is attached to the canonical device
! TypeDescriptor global.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_pointer_rtti_mod
  implicit none

  type :: base_t
    integer :: base_value
  end type base_t

  type, extends(base_t) :: child_t
    integer :: child_value
  end type child_t

end module polymorphic_pointer_rtti_mod

program main
  use polymorphic_pointer_rtti_mod
  implicit none

  type(child_t), target :: target_obj
  class(base_t), pointer :: obj
  type(base_t) :: base_mold
  logical :: extends_base
  logical :: base_extends_obj

  target_obj%base_value = 41
  target_obj%child_value = 7
  obj => target_obj

  extends_base = .false.
  base_extends_obj = .true.

  !$omp target map(tofrom: obj, extends_base, base_extends_obj) map(to: base_mold)
    extends_base = extends_type_of(obj, base_mold)
    base_extends_obj = extends_type_of(base_mold, obj)
    obj%base_value = obj%base_value + 1
  !$omp end target

  if (.not. extends_base) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (base_extends_obj) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (target_obj%base_value /= 42) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (target_obj%child_value /= 7) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
