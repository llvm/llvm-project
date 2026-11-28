! Offload test for OpenMP 6.1 polymorphic RTTI declare-target filtering.
! Mapping class(child_a_t) needs child_a_t and base_t RTTI on the device,
! but it must not mark unrelated sibling child_b_t bindings as declare
! target. If child_b_t%host_only is selected, device linking can fail
! because it references the host C library function getpid.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_no_sibling_rtti_mod
  implicit none

  interface
    integer function getpid() bind(c, name="getpid")
    end function getpid
  end interface

  type :: base_t
    integer :: v = 0
  end type base_t

  type, extends(base_t) :: child_a_t
  end type child_a_t

  type, extends(base_t) :: child_b_t
  contains
    procedure :: host_only
  end type child_b_t

contains

  subroutine host_only(this)
    class(child_b_t), intent(inout) :: this

    this%v = getpid()
  end subroutine host_only

end module target_map_polymorphic_no_sibling_rtti_mod

program main
  use target_map_polymorphic_no_sibling_rtti_mod
  implicit none

  class(child_a_t), allocatable :: a

  allocate(a)

  !$omp target map(tofrom: a)
    a%v = a%v + 1
  !$omp end target

  if (a%v /= 1) then
    print *, "======= Test Failed! ======="
    print *, a%v
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(a)
end program main

! CHECK: ======= Test Passed! =======
