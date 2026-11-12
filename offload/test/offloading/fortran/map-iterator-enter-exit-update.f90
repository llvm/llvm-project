! Offloading test for the iterator modifier on the map clauses of target
! enter/exit data and the motion clauses of target update. Each iteration maps
! one element or section, so only the selected parts may be transferred.

! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program map_iterator
  use omp_lib
  use iso_c_binding
  implicit none
  integer, parameter :: n = 8
  integer, allocatable, target :: a(:)
  integer, target :: b(n), c(n), tmp(n)
  integer :: m(4, 3), s, i, j, lo, hi, dev, host, rc
  logical :: use_device, is_present(n)

  dev = omp_get_default_device()
  host = omp_get_initial_device()

  ! Only the even elements of an allocatable array are copied back.
  allocate(a(n))
  a = [(i, i = 1, n)]
  !$omp target enter data map(to: a)
  tmp = [(100 + i, i = 1, n)]
  rc = omp_target_memcpy(omp_get_mapped_ptr(c_loc(a), dev), c_loc(tmp), &
                         int(n * c_sizeof(tmp(1)), c_size_t), 0_c_size_t, &
                         0_c_size_t, dev, host)
  a = -1
  !$omp target update from(iterator(i = 2:n:2): a(i))
  !$omp target exit data map(delete: a)
  ! CHECK: update from: -1 102 -1 104 -1 106 -1 108
  print *, "update from:", a

  ! Sections b(7:8) and b(4:5): runtime bounds, a negative step that does not
  ! divide the range, and an if clause.
  b = 0
  !$omp target enter data map(to: b)
  b = [(10 * i, i = 1, n)]
  lo = 2
  hi = 7
  use_device = .true.
  !$omp target update to(iterator(i = lo:hi): b(i)) if(.not. use_device)
  !$omp target update to(iterator(i = hi:lo:-3): b(i:i+1)) if(use_device)
  b = -1
  !$omp target update from(b)
  !$omp target exit data map(delete: b)
  ! CHECK: update to: 0 0 0 40 50 0 70 80
  print *, "update to:", b

  ! Elements 1, 3 and 5 get mappings of their own: 3 keeps the value copied by
  ! enter data, 1 and 5 are overwritten by the update.
  c = [(i, i = 1, n)]
  !$omp target enter data map(iterator(i = 1:n-2:2), to: c(i))
  is_present = [(omp_target_is_present(c_loc(c(i)), dev) /= 0, i = 1, n)]
  ! CHECK: enter data: T F T F T F F F
  print *, "enter data:", is_present
  c = [(200 + i, i = 1, n)]
  !$omp target update to(iterator(i = 1:n:4): c(i))
  c = -1
  !$omp target exit data map(iterator(i = 1:n-2:2), from: c(i))
  is_present = [(omp_target_is_present(c_loc(c(i)), dev) /= 0, i = 1, n)]
  ! CHECK: exit data: 201 -1 3 -1 205 -1 -1 -1
  print *, "exit data:", c
  ! CHECK: after exit data: F F F F F F F F
  print *, "after exit data:", is_present

  ! A static item, a 2-D iterator and a 1-D iterator in one construct.
  m = 0
  s = 0
  !$omp target enter data map(to: m, s)
  m = reshape([(i, i = 1, 12)], [4, 3])
  s = 42
  !$omp target update to(s) to(iterator(i = 1:3, j = 2:3): m(i, j)) &
  !$omp& to(iterator(i = 1:4:3): m(i, 1))
  m = -1
  s = -1
  !$omp target update from(m, s)
  !$omp target exit data map(delete: m, s)
  ! CHECK: 2-D update: 1 0 0 4 5 6 7 0 9 10 11 0 42
  print *, "2-D update:", m, s
end program map_iterator
