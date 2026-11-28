! Offload regression test for an explicit declare-target helper that uses a
! polymorphic object as data only. This should require RTTI descriptor data, but
! it should not pull unused sibling type-bound procedures into the device image.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_declare_target_data_only_mod
  implicit none

  interface
    integer function getpid() bind(c, name="getpid")
    end function getpid
  end interface

  type :: base_t
    integer :: v = 0
  end type base_t

  type, extends(base_t) :: child_t
    integer :: c = 0
  end type child_t

  type, extends(base_t) :: host_child_t
  contains
    procedure :: host_only
  end type host_child_t

  !$omp declare target(touch_data)

contains

  subroutine host_only(this)
    class(host_child_t), intent(inout) :: this

    this%v = getpid()
  end subroutine host_only

  subroutine touch_data(this)
    class(base_t), intent(inout) :: this

    this%v = this%v + 1
  end subroutine touch_data

end module target_map_polymorphic_declare_target_data_only_mod

program main
  use target_map_polymorphic_declare_target_data_only_mod
  implicit none

  class(base_t), allocatable :: obj

  allocate(child_t :: obj)
  obj%v = 1

  !$omp target map(tofrom: obj)
    call touch_data(obj)
  !$omp end target

  if (obj%v /= 2) then
    print *, "======= Test Failed! ======="
    print *, obj%v
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(obj)
end program main

! CHECK: ======= Test Passed! =======
