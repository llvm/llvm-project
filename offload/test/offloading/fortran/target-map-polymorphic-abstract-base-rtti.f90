! Offload test for polymorphic RTTI with an abstract declared type.
!
! An abstract base cannot be instantiated, so the mapped polymorphic descriptor
! has an abstract declared type and a concrete extension dynamic type.  The
! target region uses SAME_TYPE_AS and EXTENDS_TYPE_OF with a concrete child
! mold to verify device RTTI without SELECT TYPE in the target region.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_abstract_base_rtti_mod
  implicit none

  type, abstract :: abstract_base_t
    integer :: base_value
  end type abstract_base_t

  type, extends(abstract_base_t) :: child_t
    integer :: child_value
  end type child_t

end module polymorphic_abstract_base_rtti_mod

program main
  use polymorphic_abstract_base_rtti_mod
  implicit none

  class(abstract_base_t), allocatable :: obj
  type(child_t) :: child_mold
  logical :: same_child
  logical :: extends_child
  logical :: child_extends_obj

  allocate(child_t :: obj)
  obj%base_value = 8

  select type (obj)
  type is (child_t)
    obj%child_value = 34
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  same_child = .false.
  extends_child = .false.
  child_extends_obj = .false.

  !$omp target enter data map(to: obj)

  !$omp target map(tofrom: obj, same_child, extends_child, child_extends_obj) map(to: child_mold)
    same_child = same_type_as(obj, child_mold)
    extends_child = extends_type_of(obj, child_mold)
    child_extends_obj = extends_type_of(child_mold, obj)
    obj%base_value = obj%base_value + 1
  !$omp end target

  !$omp target exit data map(from: obj)

  if (.not. same_child) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (.not. extends_child) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (.not. child_extends_obj) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (obj%base_value /= 9) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
