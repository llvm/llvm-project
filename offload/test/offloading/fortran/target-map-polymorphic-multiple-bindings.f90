! Offload regression test for binding-name-specific RTTI declare-target
! marking with multiple bindings on the same dynamic type. The target region
! dispatches inc() and scale(), but not host_only(), which references getpid.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_multiple_bindings_mod
  implicit none

  interface
    integer function getpid() bind(c, name="getpid")
    end function getpid
  end interface

  type :: base_t
    integer :: n = 0
  contains
    procedure :: inc => base_inc
    procedure :: scale => base_scale
    procedure :: host_only => base_host_only
  end type base_t

  type, extends(base_t) :: child_t
    integer :: c = 0
  contains
    procedure :: inc => child_inc
    procedure :: scale => child_scale
    procedure :: host_only => child_host_only
  end type child_t

contains

  subroutine base_inc(this)
    class(base_t), intent(inout) :: this

    this%n = this%n + 1
  end subroutine base_inc

  subroutine base_scale(this)
    class(base_t), intent(inout) :: this

    this%n = this%n * 2
  end subroutine base_scale

  subroutine base_host_only(this)
    class(base_t), intent(inout) :: this

    this%n = getpid()
  end subroutine base_host_only

  subroutine child_inc(this)
    class(child_t), intent(inout) :: this

    this%n = this%n + 10
    this%c = this%c + 1
  end subroutine child_inc

  subroutine child_scale(this)
    class(child_t), intent(inout) :: this

    this%n = this%n * 3
    this%c = this%c * 5
  end subroutine child_scale

  subroutine child_host_only(this)
    class(child_t), intent(inout) :: this

    this%n = getpid()
    this%c = -1
  end subroutine child_host_only

end module target_map_polymorphic_multiple_bindings_mod

program main
  use target_map_polymorphic_multiple_bindings_mod
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
    call obj%inc()
    call obj%scale()
  !$omp end target

  failed = .false.
  if (obj%n /= 33) failed = .true.

  select type (obj)
  type is (child_t)
    if (obj%c /= 15) failed = .true.
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
