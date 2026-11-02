! Absent optional arguments represented by plain references must retain their
! absence on the device and must not be allocated or copied by their mappings.
! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-generic
! RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic 2>&1 | %fcheck-generic
! RUN: env OMP_TARGET_OFFLOAD=DISABLED %libomptarget-run-generic 2>&1 | %fcheck-generic

module optional_target_args
contains
  subroutine scalar_alloc(expected, x)
    logical, intent(in) :: expected
    integer, optional :: x
    logical :: found
    integer :: value, visits

    visits = 0
    !$omp target map(alloc:x) map(from:found,value) map(tofrom:visits)
      visits = visits + 1
      found = present(x)
      value = -1
      if (present(x)) then
        ! MAP(ALLOC:) does not initialize device storage.
        x = 73
        value = x
      endif
    !$omp end target
    if (visits /= 1 .or. (found .neqv. expected)) stop 1
    if (value /= merge(73, -1, expected)) stop 2
  end subroutine

  subroutine scalar_copy(expected, x)
    logical, intent(in) :: expected
    integer, optional :: x
    logical :: found
    integer :: value, visits

    if (present(x)) x = 42
    visits = 0
    !$omp target map(tofrom:x) map(from:found,value) map(tofrom:visits)
      visits = visits + 1
      found = present(x)
      value = -1
      if (present(x)) then
        value = x
        x = x + 1
      endif
    !$omp end target
    if (visits /= 1 .or. (found .neqv. expected)) stop 3
    if (value /= merge(42, -1, expected)) stop 4
    if (present(x)) then
      if (x /= 43) stop 5
    endif
  end subroutine

  subroutine fixed_array(expected, x)
    logical, intent(in) :: expected
    integer, optional :: x(8)
    logical :: found
    integer :: value, visits

    if (present(x)) x = 42
    visits = 0
    !$omp target map(from:found,value) map(tofrom:visits)
      visits = visits + 1
      found = present(x)
      value = -1
      if (present(x)) then
        value = x(8)
        x(8) = x(8) + 1
      endif
    !$omp end target
    if (visits /= 1 .or. (found .neqv. expected)) stop 6
    if (value /= merge(42, -1, expected)) stop 7
    if (present(x)) then
      if (x(8) /= 43 .or. any(x(1:7) /= 42)) stop 8
    endif
  end subroutine

  subroutine dynamic_array(n, expected, x)
    integer, intent(in) :: n
    logical, intent(in) :: expected
    integer, optional :: x(n)
    logical :: found
    integer :: value, visits

    if (present(x)) x = 42
    visits = 0
    !$omp target map(from:found,value) map(tofrom:visits)
      visits = visits + 1
      found = present(x)
      value = -1
      if (present(x)) then
        value = x(n)
        x(n) = x(n) + 1
      endif
    !$omp end target
    if (visits /= 1 .or. (found .neqv. expected)) stop 9
    if (value /= merge(42, -1, expected)) stop 10
    if (present(x)) then
      if (x(n) /= 43 .or. any(x(1:n-1) /= 42)) stop 11
    endif
  end subroutine

  subroutine array_alloc(n, expected, x)
    integer, intent(in) :: n
    logical, intent(in) :: expected
    integer, optional :: x(n)
    logical :: found
    integer :: value, visits

    visits = 0
    !$omp target map(alloc:x) map(from:found,value) map(tofrom:visits)
      visits = visits + 1
      found = present(x)
      value = -1
      if (present(x)) then
        x(n) = 73
        value = x(n)
      endif
    !$omp end target
    if (visits /= 1 .or. (found .neqv. expected)) stop 12
    if (value /= merge(73, -1, expected)) stop 13
  end subroutine

  subroutine array_section(n, m, expected, x)
    integer, intent(in) :: n, m
    logical, intent(in) :: expected
    integer, optional :: x(n, m)
    logical :: found
    integer :: value, visits

    if (present(x)) x = 42
    visits = 0
    !$omp target map(tofrom:x(1:n,2:3),visits) map(from:found,value)
      visits = visits + 1
      found = present(x)
      value = -1
      if (present(x)) then
        value = x(1,2) + x(n,3)
        x(1,2) = 43
        x(n,3) = 73
      endif
    !$omp end target
    if (visits /= 1 .or. (found .neqv. expected)) stop 14
    if (value /= merge(84, -1, expected)) stop 15
    if (present(x)) then
      if (x(1,2) /= 43 .or. x(n,3) /= 73) stop 16
      if (x(1,1) /= 42 .or. x(n,m) /= 42 .or. x(2,2) /= 42) stop 17
    endif
  end subroutine

  subroutine forward(n, expected, scalar, array, matrix)
    integer, intent(in) :: n
    logical, intent(in) :: expected
    integer, optional :: scalar, array(n), matrix(5,4)
    call scalar_alloc(expected, scalar)
    call scalar_copy(expected, scalar)
    call fixed_array(expected, array)
    call dynamic_array(n, expected, array)
    call array_alloc(n, expected, array)
    call array_section(5, 4, expected, matrix)
  end subroutine
end module

program main
  use optional_target_args
  integer :: scalar, array(8), matrix(5,4)

  call scalar_alloc(.false.)
  call scalar_alloc(.true., scalar)
  call scalar_copy(.false.)
  call scalar_copy(.true., scalar)
  call fixed_array(.false.)
  call fixed_array(.true., array)
  call dynamic_array(8, .false.)
  call dynamic_array(8, .true., array)
  call array_alloc(8, .false.)
  call array_alloc(8, .true., array)
  call array_section(5, 4, .false.)
  call array_section(5, 4, .true., matrix)
  call forward(8, .false.)
  call forward(8, .true., scalar, array, matrix)
  print *, "PASS"
end program

! CHECK: PASS
