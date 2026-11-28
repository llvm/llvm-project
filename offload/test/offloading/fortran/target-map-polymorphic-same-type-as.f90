! Offload test for SAME_TYPE_AS using mapped polymorphic Fortran
! descriptors without SELECT TYPE in the target region.
!
! SAME_TYPE_AS is stricter than EXTENDS_TYPE_OF and requires exact dynamic type
! identity for polymorphic descriptor RTTI on device.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_same_type_as_mod
  implicit none

  type :: base_t
    integer :: base_value
  end type base_t

  type, extends(base_t) :: child_t
    integer :: child_value
  end type child_t

end module polymorphic_same_type_as_mod

program main
  use polymorphic_same_type_as_mod
  implicit none

  class(base_t), allocatable :: a
  class(base_t), allocatable :: b
  type(base_t) :: base_mold
  type(child_t) :: child_mold
  logical :: same_self
  logical :: same_other_child
  logical :: same_base
  logical :: same_child_mold

  allocate(child_t :: a)
  allocate(child_t :: b)
  a%base_value = 1
  b%base_value = 2

  same_self = .false.
  same_other_child = .false.
  same_base = .true.
  same_child_mold = .false.

  !$omp target enter data map(to: a, b)

  !$omp target map(tofrom: a, b, same_self, same_other_child, same_base, same_child_mold) map(to: base_mold, child_mold)
    same_self = same_type_as(a, a)
    same_other_child = same_type_as(a, b)
    same_base = same_type_as(a, base_mold)
    same_child_mold = same_type_as(a, child_mold)
    a%base_value = a%base_value + 1
    b%base_value = b%base_value + 1
  !$omp end target

  !$omp target exit data map(from: a, b)

  if (.not. same_self) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (.not. same_other_child) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (same_base) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (.not. same_child_mold) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (a%base_value /= 2) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (b%base_value /= 3) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
