! Offload test for a polymorphic descriptor nested inside a derived type.
!
! This verifies that descriptor addendum RTTI handling still works when the
! descriptor is a derived-type component instead of the top-level map item.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_nested_rtti_mod
  implicit none

  type :: base_t
    integer :: base_value
  end type base_t

  type, extends(base_t) :: child_t
    integer :: child_value
  end type child_t

  type :: wrapper_t
    integer :: marker
    class(base_t), allocatable :: item
  end type wrapper_t

contains

  subroutine init_wrapper(w)
    type(wrapper_t), intent(out) :: w

    w%marker = 5
    allocate(child_t :: w%item)
    w%item%base_value = 12
  end subroutine init_wrapper

end module polymorphic_nested_rtti_mod

program main
  use polymorphic_nested_rtti_mod
  implicit none

  type(wrapper_t) :: w
  type(base_t) :: base_mold
  logical :: extends_base
  logical :: base_extends_item

  call init_wrapper(w)

  extends_base = .false.
  base_extends_item = .true.

  !$omp target map(tofrom: w, extends_base, base_extends_item) map(to: base_mold)
    extends_base = extends_type_of(w%item, base_mold)
    base_extends_item = extends_type_of(base_mold, w%item)
    w%marker = w%marker + 1
    w%item%base_value = w%item%base_value + 1
  !$omp end target

  if (.not. extends_base) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (base_extends_item) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (w%marker /= 6) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (w%item%base_value /= 13) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
