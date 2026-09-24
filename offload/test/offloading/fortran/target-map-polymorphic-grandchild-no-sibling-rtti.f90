! Offload regression test for polymorphic RTTI filtering with a grandchild.
! Mapping class(child_a_t) may require grandchild_a_t RTTI, but it must not
! select unrelated sibling child_b_t bindings.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_grandchild_no_sibling_rtti_mod
  implicit none

  interface
    integer function getpid() bind(c, name="getpid")
    end function getpid
  end interface

  type :: base_t
    integer :: v = 0
  end type base_t

  type, extends(base_t) :: child_a_t
    integer :: a = 0
  end type child_a_t

  type, extends(child_a_t) :: grandchild_a_t
    integer :: g = 0
  end type grandchild_a_t

  type, extends(base_t) :: child_b_t
  contains
    procedure :: host_only
  end type child_b_t

contains

  subroutine host_only(this)
    class(child_b_t), intent(inout) :: this

    this%v = getpid()
  end subroutine host_only

end module target_map_polymorphic_grandchild_no_sibling_rtti_mod

program main
  use target_map_polymorphic_grandchild_no_sibling_rtti_mod
  implicit none

  class(child_a_t), allocatable :: a
  logical :: failed

  allocate(grandchild_a_t :: a)
  a%v = 1
  a%a = 2
  select type (a)
  type is (grandchild_a_t)
    a%g = 3
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  !$omp target map(tofrom: a)
    a%v = a%v + 10
    a%a = a%a + 20
    select type (a)
    type is (grandchild_a_t)
      a%g = a%g + 30
    class default
      a%v = -1
    end select
  !$omp end target

  failed = .false.
  if (a%v /= 11) failed = .true.
  if (a%a /= 22) failed = .true.
  select type (a)
  type is (grandchild_a_t)
    if (a%g /= 33) failed = .true.
  class default
    failed = .true.
  end select

  if (failed) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(a)
end program main

! CHECK: ======= Test Passed! =======
