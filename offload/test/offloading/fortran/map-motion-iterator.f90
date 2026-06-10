! Offloading test for iterator modifier on map and motion clauses.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program map_motion_iterator
  implicit none
  integer, parameter :: n = 8
  integer :: a(n), b(n), c(n), d(n), e(n), f(n), g(n), h(n)
  integer :: i, dyn_n, dyn_step
  logical :: use_device

  do i = 1, n
    a(i) = i
  end do
  !$omp target enter data map(to: a)
  !$omp target
    do i = 2, n, 2
      a(i) = 100 + i
    end do
  !$omp end target
  do i = 2, n, 2
    a(i) = -100
  end do
  !$omp target update from(iterator(i = 2:n:2): a(i))
  !$omp target exit data map(delete: a)

  ! CHECK: update from: 1 102 3 104 5 106 7 108
  print *, "update from:", a

  do i = 1, n
    b(i) = i
  end do
  !$omp target enter data map(to: b)
  do i = 2, n, 2
    b(i) = 200 + i
  end do
  !$omp target update to(iterator(i = 2:n:2): b(i))
  !$omp target
    do i = 2, n, 2
      b(i) = b(i) + 10
    end do
  !$omp end target
  do i = 2, n, 2
    b(i) = -100
  end do
  !$omp target update from(iterator(i = 2:n:2): b(i))
  !$omp target exit data map(delete: b)

  ! CHECK: update tofrom: 1 212 3 214 5 216 7 218
  print *, "update tofrom:", b

  do i = 1, n
    c(i) = i
  end do
  !$omp target enter data map(to: c)
  !$omp target
    c(1) = 11
    c(3) = 33
    c(5) = 55
    c(7) = 77
  !$omp end target
  do i = 1, n, 2
    c(i) = -100
  end do
  !$omp target exit data map(iterator(i = 1:n:2), from: c(i))
  !$omp target exit data map(delete: c)

  ! CHECK: exit data: 11 2 33 4 55 6 77 8
  print *, "exit data:", c

  do i = 1, n
    d(i) = i
  end do
  !$omp target data map(iterator(i = 1:n:2), tofrom: d(i))
    !$omp target map(present, alloc: d(1))
      d(1) = 21
    !$omp end target
    !$omp target map(present, alloc: d(3))
      d(3) = 43
    !$omp end target
    !$omp target map(present, alloc: d(5))
      d(5) = 65
    !$omp end target
    !$omp target map(present, alloc: d(7))
      d(7) = 87
    !$omp end target
  !$omp end target data

  ! CHECK: target data: 21 2 43 4 65 6 87 8
  print *, "target data:", d

  dyn_n = n
  dyn_step = 2
  use_device = .true.
  do i = 1, n
    e(i) = i
  end do
  !$omp target data if(use_device) &
  !$omp& map(iterator(i = 1:dyn_n:dyn_step), tofrom: e(i))
    !$omp target map(present, alloc: e(1))
      e(1) = 31
    !$omp end target
    !$omp target map(present, alloc: e(3))
      e(3) = 53
    !$omp end target
    !$omp target map(present, alloc: e(5))
      e(5) = 75
    !$omp end target
    !$omp target map(present, alloc: e(7))
      e(7) = 97
    !$omp end target
  !$omp end target data

  ! CHECK: target data if: 31 2 53 4 75 6 97 8
  print *, "target data if:", e

  ! nowait: target update from with iterator modifier.
  do i = 1, n
    f(i) = i
  end do
  !$omp target enter data map(to: f)
  !$omp target
    do i = 2, n, 2
      f(i) = 300 + i
    end do
  !$omp end target
  do i = 2, n, 2
    f(i) = -100
  end do
  !$omp target update from(iterator(i = 2:n:2): f(i)) nowait
  !$omp taskwait
  !$omp target exit data map(delete: f)

  ! CHECK: update from nowait: 1 302 3 304 5 306 7 308
  print *, "update from nowait:", f

  ! nowait: target enter data + target update to with iterator modifier.
  do i = 1, n
    g(i) = i
  end do
  !$omp target enter data map(iterator(i = 1:n:2), to: g(i)) nowait
  !$omp taskwait
  do i = 1, n, 2
    g(i) = 400 + i
  end do
  !$omp target update to(iterator(i = 1:n:2): g(i)) nowait
  !$omp taskwait
  !$omp target map(present, alloc: g(1)) map(present, alloc: g(3)) &
  !$omp&       map(present, alloc: g(5)) map(present, alloc: g(7))
    g(1) = g(1) + 1
    g(3) = g(3) + 1
    g(5) = g(5) + 1
    g(7) = g(7) + 1
  !$omp end target
  do i = 1, n, 2
    g(i) = -100
  end do
  !$omp target update from(iterator(i = 1:n:2): g(i)) nowait
  !$omp taskwait
  !$omp target exit data map(delete: g)

  ! CHECK: enter/update nowait: 402 2 404 4 406 6 408 8
  print *, "enter/update nowait:", g

  ! nowait: target exit data from with iterator modifier.
  do i = 1, n
    h(i) = i
  end do
  !$omp target enter data map(to: h)
  !$omp target
    do i = 1, n, 2
      h(i) = 500 + i
    end do
  !$omp end target
  do i = 1, n, 2
    h(i) = -100
  end do
  !$omp target exit data map(iterator(i = 1:n:2), from: h(i)) nowait
  !$omp taskwait

  ! CHECK: exit data nowait: 501 2 503 4 505 6 507 8
  print *, "exit data nowait:", h

end program map_motion_iterator

