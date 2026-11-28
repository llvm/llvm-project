! Offload regression test for polymorphic dispatch hidden behind a helper
! routine called from a target region. The target body calls helper, and helper
! performs the type-bound dispatch, so declare-target processing must still make
! the dispatch binding available on the device.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_helper_dispatch_mod
  implicit none

  type :: base_t
    integer :: v = 0
  contains
    procedure :: bump => base_bump
  end type base_t

  type, extends(base_t) :: child_t
    integer :: c = 0
  contains
    procedure :: bump => child_bump
  end type child_t

contains

  subroutine base_bump(this)
    class(base_t), intent(inout) :: this

    this%v = this%v + 1
  end subroutine base_bump

  subroutine child_bump(this)
    class(child_t), intent(inout) :: this

    this%v = this%v + 10
    this%c = this%c + 1
  end subroutine child_bump

  subroutine helper(this)
    class(base_t), intent(inout) :: this

    call this%bump()
  end subroutine helper

end module target_map_polymorphic_helper_dispatch_mod

program main
  use target_map_polymorphic_helper_dispatch_mod
  implicit none

  class(base_t), allocatable :: obj
  logical :: failed

  allocate(child_t :: obj)
  obj%v = 1
  select type (obj)
  type is (child_t)
    obj%c = 2
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  !$omp target map(tofrom: obj)
    call helper(obj)
  !$omp end target

  failed = .false.
  if (obj%v /= 11) failed = .true.
  select type (obj)
  type is (child_t)
    if (obj%c /= 3) failed = .true.
  class default
    failed = .true.
  end select

  if (failed) then
    print *, "======= Test Failed! ======="
    print *, obj%v
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(obj)
end program main

! CHECK: ======= Test Passed! =======
