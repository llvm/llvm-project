! Offload regression test ensuring ancestor processing still keeps inherited
! type-bound procedures available on the device.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_inherited_binding_rtti_mod
  implicit none

  type :: base_t
    integer :: v = 0
  contains
    procedure :: inc
  end type base_t

  type, extends(base_t) :: child_t
    integer :: c = 0
  end type child_t

contains

  subroutine inc(this, amount)
    class(base_t), intent(inout) :: this
    integer, intent(in) :: amount

    this%v = this%v + amount
  end subroutine inc

end module target_map_polymorphic_inherited_binding_rtti_mod

program main
  use target_map_polymorphic_inherited_binding_rtti_mod
  implicit none

  class(child_t), allocatable :: a
  logical :: failed

  allocate(child_t :: a)
  a%v = 1
  a%c = 2

  !$omp target map(tofrom: a)
    call a%inc(10)
    a%c = a%c + 20
  !$omp end target

  failed = .false.
  if (a%v /= 11) failed = .true.
  if (a%c /= 22) failed = .true.

  if (failed) then
    print *, "======= Test Failed! ======="
    print *, a%v, a%c
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(a)
end program main

! CHECK: ======= Test Passed! =======
