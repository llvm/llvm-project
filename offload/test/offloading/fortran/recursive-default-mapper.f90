! Offloading test for recursive default mapper emission
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

module recursive_mapper_mod
  implicit none

  type :: inner
    integer :: value
    type(inner), pointer :: next
  end type inner

  type :: outer
    integer, allocatable :: arr(:)
    type(inner), pointer :: head
  end type outer

contains

end module recursive_mapper_mod

program main
  use recursive_mapper_mod
  implicit none

  type(outer) :: o

  allocate(o%arr(2))
  o%arr = [1, 2]

  !$omp target map(tofrom: o)
    o%arr(1) = o%arr(1) + 1
    o%arr(2) = o%arr(2) + 1
  !$omp end target

  print *, o%arr(1), o%arr(2)
end program main

! CHECK: 2 3
