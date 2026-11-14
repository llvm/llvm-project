! Offload test for dynamic dispatch through a mapped polymorphic descriptor
! with multiple levels of type extension.
!
! This verifies that the device RTTI binding table for the most-derived dynamic
! type is used when dispatching through a base-class descriptor, including both
! function and subroutine type-bound procedures.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_multilevel_type_bound_dispatch_mod
  implicit none

  type :: base_t
    integer :: base_value
  contains
    procedure :: value => base_value_fn
    procedure :: bump => base_bump
  end type base_t

  type, extends(base_t) :: middle_t
    integer :: middle_value
  contains
    procedure :: value => middle_value_fn
    procedure :: bump => middle_bump
  end type middle_t

  type, extends(middle_t) :: leaf_t
    integer :: leaf_value
  contains
    procedure :: value => leaf_value_fn
    procedure :: bump => leaf_bump
  end type leaf_t

contains

  integer function base_value_fn(self, scale)
    class(base_t), intent(in) :: self
    integer, intent(in) :: scale
    base_value_fn = self%base_value * scale
  end function base_value_fn

  subroutine base_bump(self, amount)
    class(base_t), intent(inout) :: self
    integer, intent(in) :: amount
    self%base_value = self%base_value + amount
  end subroutine base_bump

  integer function middle_value_fn(self, scale)
    class(middle_t), intent(in) :: self
    integer, intent(in) :: scale
    middle_value_fn = (self%base_value + self%middle_value) * scale
  end function middle_value_fn

  subroutine middle_bump(self, amount)
    class(middle_t), intent(inout) :: self
    integer, intent(in) :: amount
    self%base_value = self%base_value + amount
    self%middle_value = self%middle_value + amount * 2
  end subroutine middle_bump

  integer function leaf_value_fn(self, scale)
    class(leaf_t), intent(in) :: self
    integer, intent(in) :: scale
    leaf_value_fn = (self%base_value + self%middle_value + self%leaf_value) * scale
  end function leaf_value_fn

  subroutine leaf_bump(self, amount)
    class(leaf_t), intent(inout) :: self
    integer, intent(in) :: amount
    self%base_value = self%base_value + amount
    self%middle_value = self%middle_value + amount * 2
    self%leaf_value = self%leaf_value + amount * 3
  end subroutine leaf_bump

end module polymorphic_multilevel_type_bound_dispatch_mod

program main
  use polymorphic_multilevel_type_bound_dispatch_mod
  implicit none

  class(base_t), allocatable :: obj
  integer :: before
  integer :: after

  allocate(leaf_t :: obj)
  obj%base_value = 2

  select type (obj)
  type is (leaf_t)
    obj%middle_value = 3
    obj%leaf_value = 5
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  before = -1
  after = -1

  select type (obj)
  type is (leaf_t)
    !$omp target enter data map(to: obj)
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  !$omp target map(tofrom: obj, before, after)
    before = obj%value(4)
    call obj%bump(2)
    after = obj%value(1)
  !$omp end target

  select type (obj)
  type is (leaf_t)
    !$omp target exit data map(from: obj)
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  if (before /= 40) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (after /= 22) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  select type (obj)
  type is (leaf_t)
    if (obj%base_value /= 4) then
      print *, "======= Test Failed! ======="
      stop 1
    end if
    if (obj%middle_value /= 7) then
      print *, "======= Test Failed! ======="
      stop 1
    end if
    if (obj%leaf_value /= 11) then
      print *, "======= Test Failed! ======="
      stop 1
    end if
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
