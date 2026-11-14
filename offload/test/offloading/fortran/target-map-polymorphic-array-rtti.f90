! Offload test for polymorphic Fortran array descriptor RTTI without
! SELECT TYPE in the target region.
!
! This exercises the array descriptor path while using EXTENDS_TYPE_OF on a
! polymorphic array element to validate derived-type RTTI on device.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_array_rtti_mod
  implicit none

  type :: base_t
    integer :: base_value
  end type base_t

  type, extends(base_t) :: child_t
    integer :: child_value
  end type child_t

end module polymorphic_array_rtti_mod

program main
  use polymorphic_array_rtti_mod
  implicit none

  class(base_t), allocatable :: arr(:)
  type(base_t) :: base_mold
  logical :: extends_base
  logical :: base_extends_elem

  allocate(child_t :: arr(1))
  arr(1)%base_value = 3

  extends_base = .false.
  base_extends_elem = .true.

  !$omp target enter data map(to: arr)

  !$omp target map(tofrom: arr, extends_base, base_extends_elem) map(to: base_mold)
    extends_base = extends_type_of(arr(1), base_mold)
    base_extends_elem = extends_type_of(base_mold, arr(1))
    arr(1)%base_value = arr(1)%base_value + 10
  !$omp end target

  !$omp target exit data map(from: arr)

  if (.not. extends_base) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (base_extends_elem) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (arr(1)%base_value /= 13) then
    print *, "======= Test Failed! ======="
    stop 1
  end if
  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
