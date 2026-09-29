! This test validates that declare mapper for a derived type that extends
! a parent type with an allocatable component correctly maps the nested
! allocatable payload via the mapper when the whole object is mapped on
! target.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_parent_allocatable
  implicit none

  type, abstract :: base_t
    real, allocatable :: base_arr(:)
  end type base_t

  type, extends(base_t) :: real_t
    real, allocatable :: real_arr(:)
  end type real_t
  !$omp declare mapper(custommapper: real_t :: t) map(t%base_arr, t%real_arr)

  type(real_t) :: r
  integer :: i
  allocate(r%base_arr(10), source=1.0)
  allocate(r%real_arr(10), source=1.0)

  !$omp target map(mapper(custommapper), tofrom: r)
  do i = 1, size(r%base_arr)
    r%base_arr(i) = 2.0
    r%real_arr(i) = 3.0
    r%real_arr(i) = r%base_arr(1)
  end do
  !$omp end target


  !CHECK: base_arr:  2. 2. 2. 2. 2. 2. 2. 2. 2. 2.
  print*, "base_arr: ", r%base_arr
  !CHECK: real_arr:  2. 2. 2. 2. 2. 2. 2. 2. 2. 2.
  print*, "real_arr: ", r%real_arr

  deallocate(r%real_arr)
  deallocate(r%base_arr)
end program target_declare_mapper_parent_allocatable
