! Offload test for dynamic dispatch through a mapped polymorphic descriptor
! whose declared type is abstract and whose binding is deferred.
!
! This verifies that device dispatch does not need a concrete implementation on
! the static abstract type.  Instead, the mapped descriptor's device RTTI must
! identify the concrete dynamic type and route the deferred binding call to that
! concrete type's device procedure.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_abstract_type_bound_dispatch_mod
  implicit none

  type, abstract :: shape_t
    integer :: bias
  contains
    procedure(value_iface), deferred :: value
    procedure(update_iface), deferred :: update
  end type shape_t

  abstract interface
    integer function value_iface(self, scale)
      import :: shape_t
      class(shape_t), intent(in) :: self
      integer, intent(in) :: scale
    end function value_iface

    subroutine update_iface(self, amount)
      import :: shape_t
      class(shape_t), intent(inout) :: self
      integer, intent(in) :: amount
    end subroutine update_iface
  end interface

  type, extends(shape_t) :: rectangle_t
    integer :: width
    integer :: height
  contains
    procedure :: value => rectangle_value
    procedure :: update => rectangle_update
  end type rectangle_t

contains

  integer function rectangle_value(self, scale)
    class(rectangle_t), intent(in) :: self
    integer, intent(in) :: scale
    rectangle_value = (self%width * self%height + self%bias) * scale
  end function rectangle_value

  subroutine rectangle_update(self, amount)
    class(rectangle_t), intent(inout) :: self
    integer, intent(in) :: amount
    self%bias = self%bias + amount
    self%width = self%width + 1
    self%height = self%height + 2
  end subroutine rectangle_update

end module polymorphic_abstract_type_bound_dispatch_mod

program main
  use polymorphic_abstract_type_bound_dispatch_mod
  implicit none

  class(shape_t), allocatable :: obj
  integer :: before
  integer :: after

  allocate(rectangle_t :: obj)
  obj%bias = 1

  select type (obj)
  type is (rectangle_t)
    obj%width = 4
    obj%height = 5
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  before = -1
  after = -1

  !$omp target enter data map(to: obj)

  !$omp target map(tofrom: obj, before, after)
    before = obj%value(2)
    call obj%update(3)
    after = obj%value(1)
  !$omp end target

  !$omp target exit data map(from: obj)

  if (before /= 42) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (after /= 39) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  select type (obj)
  type is (rectangle_t)
    if (obj%bias /= 4) then
      print *, "======= Test Failed! ======="
      stop 1
    end if
    if (obj%width /= 5) then
      print *, "======= Test Failed! ======="
      stop 1
    end if
    if (obj%height /= 7) then
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
