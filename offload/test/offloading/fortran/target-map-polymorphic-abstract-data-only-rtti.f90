! Offload regression test for data-only mapping through an abstract declared
! type. The dynamic type is concrete_good_t, and an unrelated concrete_bad_t has
! a host-only binding that must not be selected for a data-only target region.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_abstract_data_only_rtti_mod
  implicit none

  interface
    integer function getpid() bind(c, name="getpid")
    end function getpid
  end interface

  type, abstract :: abs_t
    integer :: n = 0
  end type abs_t

  type, extends(abs_t) :: concrete_good_t
    integer :: good = 0
  end type concrete_good_t

  type, extends(abs_t) :: concrete_bad_t
  contains
    procedure :: host_only
  end type concrete_bad_t

contains

  subroutine host_only(this)
    class(concrete_bad_t), intent(inout) :: this

    this%n = getpid()
  end subroutine host_only

end module target_map_polymorphic_abstract_data_only_rtti_mod

program main
  use target_map_polymorphic_abstract_data_only_rtti_mod
  implicit none

  class(abs_t), allocatable :: obj
  logical :: failed

  allocate(concrete_good_t :: obj)
  obj%n = 1
  select type (obj)
  type is (concrete_good_t)
    obj%good = 2
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  !$omp target map(tofrom: obj)
    obj%n = obj%n + 1
    select type (obj)
    type is (concrete_good_t)
      obj%good = obj%good + 1
    class default
      obj%n = -1
    end select
  !$omp end target

  failed = .false.
  if (obj%n /= 2) failed = .true.
  select type (obj)
  type is (concrete_good_t)
    if (obj%good /= 3) failed = .true.
  class default
    failed = .true.
  end select

  if (failed) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(obj)
end program main

! CHECK: ======= Test Passed! =======
