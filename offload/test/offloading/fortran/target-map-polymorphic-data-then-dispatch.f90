! Offload regression test for seeing the same polymorphic type first through a
! data-only map and later through an actual polymorphic dispatch. The RTTI pass
! must revisit the type with binding requirements when the later dispatch is
! discovered.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_data_then_dispatch_mod
  implicit none

  type :: base_t
    integer :: n = 0
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

    this%n = this%n + 1
  end subroutine base_bump

  subroutine child_bump(this)
    class(child_t), intent(inout) :: this

    this%n = this%n + 10
    this%c = this%c + 1
  end subroutine child_bump

end module target_map_polymorphic_data_then_dispatch_mod

program main
  use target_map_polymorphic_data_then_dispatch_mod
  implicit none

  class(base_t), allocatable :: obj
  logical :: failed

  allocate(child_t :: obj)
  obj%n = 1
  select type (obj)
  type is (child_t)
    obj%c = 2
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  !$omp target map(tofrom: obj)
    obj%n = obj%n + 1
  !$omp end target

  !$omp target map(tofrom: obj)
    call obj%bump()
  !$omp end target

  failed = .false.
  if (obj%n /= 12) failed = .true.
  select type (obj)
  type is (child_t)
    if (obj%c /= 3) failed = .true.
  class default
    failed = .true.
  end select

  if (failed) then
    print *, "======= Test Failed! ======="
    print *, obj%n
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(obj)
end program main

! CHECK: ======= Test Passed! =======
