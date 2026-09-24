! Offload regression test for binding-name precision. The target region
! dispatches only used(), while the same dynamic type also has an unrelated
! host_only() binding that references getpid.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_unused_binding_name_mod
  implicit none

  interface
    integer function getpid() bind(c, name="getpid")
    end function getpid
  end interface

  type :: base_t
    integer :: v = 0
  contains
    procedure :: used => base_used
    procedure :: host_only => base_host_only
  end type base_t

  type, extends(base_t) :: child_t
    integer :: c = 0
  contains
    procedure :: used => child_used
    procedure :: host_only => child_host_only
  end type child_t

contains

  subroutine base_used(this)
    class(base_t), intent(inout) :: this

    this%v = this%v + 1
  end subroutine base_used

  subroutine base_host_only(this)
    class(base_t), intent(inout) :: this

    this%v = getpid()
  end subroutine base_host_only

  subroutine child_used(this)
    class(child_t), intent(inout) :: this

    this%v = this%v + 10
    this%c = this%c + 1
  end subroutine child_used

  subroutine child_host_only(this)
    class(child_t), intent(inout) :: this

    this%v = getpid()
    this%c = -1
  end subroutine child_host_only

end module target_map_polymorphic_unused_binding_name_mod

program main
  use target_map_polymorphic_unused_binding_name_mod
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
    call obj%used()
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
