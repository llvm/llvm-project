! Offloading test checking that privatization of allocatables and arrays works.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

module privatization_mod
  use omp_lib
  implicit none
  ! We set large enough launch bounds to ensure support is properly tested.
  ! Lower numbers of threads/teams can hide bugs when scratch memory is used
  ! where shared memory should be used instead.
  integer, parameter :: NUM_THREADS = 128
  integer, parameter :: MIN_THREADS = 64
  integer, parameter :: NUM_TEAMS = 64
  integer, parameter :: VALUE = 12345
contains
  subroutine target_parallel_scalar(seen, nthr)
    integer, intent(out) :: seen(0:NUM_THREADS-1), nthr
    integer, allocatable :: x
    integer :: tid
    allocate(x)
    !$omp target map(from: seen, nthr) private(x) thread_limit(NUM_THREADS)
      x = VALUE
      !$omp parallel num_threads(NUM_THREADS) private(tid)
        tid = omp_get_thread_num()
        if (tid == 0) nthr = omp_get_num_threads()
        if (tid < NUM_THREADS) seen(tid) = x
      !$omp end parallel
    !$omp end target
    deallocate(x)
  end subroutine

  subroutine target_parallel_array(seen, nthr)
    integer, intent(out) :: seen(0:NUM_THREADS-1), nthr
    integer :: x(4)
    integer :: tid
    !$omp target map(from: seen, nthr) private(x) thread_limit(NUM_THREADS)
      x(3) = VALUE
      !$omp parallel num_threads(NUM_THREADS) private(tid)
        tid = omp_get_thread_num()
        if (tid == 0) nthr = omp_get_num_threads()
        if (tid < NUM_THREADS) seen(tid) = x(3)
      !$omp end parallel
    !$omp end target
  end subroutine

  subroutine target_parallel_variable_array(seen, nthr, sz)
    integer, intent(out) :: seen(0:NUM_THREADS-1), nthr
    integer, intent(in) :: sz
    integer :: x(sz)
    integer :: tid
    !$omp target map(from: seen, nthr) private(x) thread_limit(NUM_THREADS)
      x(3) = VALUE
      !$omp parallel num_threads(NUM_THREADS) private(tid)
        tid = omp_get_thread_num()
        if (tid == 0) nthr = omp_get_num_threads()
        if (tid < NUM_THREADS) seen(tid) = x(3)
      !$omp end parallel
    !$omp end target
  end subroutine

  subroutine target_parallel_allocatable_array(seen, nthr)
    integer, intent(out) :: seen(0:NUM_THREADS-1), nthr
    integer, allocatable :: x(:)
    integer :: tid
    allocate(x(4))
    !$omp target map(from: seen, nthr) private(x) thread_limit(NUM_THREADS)
      x(3) = VALUE
      !$omp parallel num_threads(NUM_THREADS) private(tid)
        tid = omp_get_thread_num()
        if (tid == 0) nthr = omp_get_num_threads()
        if (tid < NUM_THREADS) seen(tid) = x(3)
      !$omp end parallel
    !$omp end target
    deallocate(x)
  end subroutine

  subroutine target_teams_distribute_parallel_scalar(seen, nthr)
    integer, intent(out) :: seen(0:NUM_TEAMS-1, 0:NUM_THREADS-1), nthr
    integer, allocatable :: x, y
    integer :: tid, i
    allocate(x, y)
    !$omp target teams map(from: seen, nthr) private(x) &
    !$omp& thread_limit(NUM_THREADS) num_teams(NUM_TEAMS)
      x = VALUE
      !$omp distribute private(y)
      do i = 1, NUM_TEAMS
        y = VALUE
        !$omp parallel num_threads(NUM_THREADS) private(tid)
          tid = omp_get_thread_num()
          if (tid == 0) nthr = omp_get_num_threads()
          if (tid < NUM_THREADS) seen(i-1, tid) = x + y
        !$omp end parallel
      end do
    !$omp end target teams
    deallocate(x, y)
  end subroutine

  subroutine target_teams_distribute_parallel_array(seen, nthr)
    integer, intent(out) :: seen(0:NUM_TEAMS-1, 0:NUM_THREADS-1), nthr
    integer :: x(4), y(4)
    integer :: tid, i
    !$omp target teams map(from: seen, nthr) private(x) &
    !$omp& thread_limit(NUM_THREADS) num_teams(NUM_TEAMS)
      x(3) = VALUE
      !$omp distribute private(y)
      do i = 1, NUM_TEAMS
        y(3) = VALUE
        !$omp parallel num_threads(NUM_THREADS) private(tid)
          tid = omp_get_thread_num()
          if (tid == 0) nthr = omp_get_num_threads()
          if (tid < NUM_THREADS) seen(i-1, tid) = x(3) + y(3)
        !$omp end parallel
      end do
    !$omp end target teams
  end subroutine

  subroutine target_teams_distribute_parallel_variable_array(seen, nthr, sz)
    integer, intent(out) :: seen(0:NUM_TEAMS-1, 0:NUM_THREADS-1), nthr
    integer, intent(in) :: sz
    integer :: x(sz), y(sz)
    integer :: tid, i
    !$omp target teams map(from: seen, nthr) private(x) &
    !$omp& thread_limit(NUM_THREADS) num_teams(NUM_TEAMS)
      x(3) = VALUE
      !$omp distribute private(y)
      do i = 1, NUM_TEAMS
        y(3) = VALUE
        !$omp parallel num_threads(NUM_THREADS) private(tid)
          tid = omp_get_thread_num()
          if (tid == 0) nthr = omp_get_num_threads()
          if (tid < NUM_THREADS) seen(i-1, tid) = x(3) + y(3)
        !$omp end parallel
      end do
    !$omp end target teams
  end subroutine

  subroutine target_teams_distribute_parallel_allocatable_array(seen, nthr)
    integer, intent(out) :: seen(0:NUM_TEAMS-1, 0:NUM_THREADS-1), nthr
    integer, allocatable :: x(:), y(:)
    integer :: tid, i
    allocate(x(4), y(4))
    !$omp target teams map(from: seen, nthr) private(x) &
    !$omp& thread_limit(NUM_THREADS) num_teams(NUM_TEAMS)
      x(3) = VALUE
      !$omp distribute private(y)
      do i = 1, NUM_TEAMS
        y(3) = VALUE
        !$omp parallel num_threads(NUM_THREADS) private(tid)
          tid = omp_get_thread_num()
          if (tid == 0) nthr = omp_get_num_threads()
          if (tid < NUM_THREADS) seen(i-1, tid) = x(3) + y(3)
        !$omp end parallel
      end do
    !$omp end target teams
    deallocate(x, y)
  end subroutine
end module

program main
  use privatization_mod
  implicit none
  integer :: nthr, i, j
  integer :: seen1d(0:NUM_THREADS-1), seen2d(0:NUM_TEAMS-1, 0:NUM_THREADS-1)

  ! CHECK: SUCCESS: target_parallel_scalar
  seen1d = -1
  nthr = 0
  call target_parallel_scalar(seen1d, nthr)
  do i = 1, nthr
    if (seen1d(i-1) /= VALUE) then
      print *, "ERROR: target_parallel_scalar (unexpected results)"
    end if
  end do
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: target_parallel_scalar (too few threads)"
  else
    print *, "SUCCESS: target_parallel_scalar"
  end if

  ! CHECK: SUCCESS: target_parallel_array
  seen1d = -1
  nthr = 0
  call target_parallel_array(seen1d, nthr)
  do i = 1, nthr
    if (seen1d(i-1) /= VALUE) then
      print *, "ERROR: target_parallel_array (unexpected results)"
      stop 1
    end if
  end do
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: target_parallel_array (too few threads)"
  else
    print *, "SUCCESS: target_parallel_array"
  end if

  ! CHECK: SUCCESS: target_parallel_variable_array
  seen1d = -1
  nthr = 0
  call target_parallel_variable_array(seen1d, nthr, 4)
  do i = 1, nthr
    if (seen1d(i-1) /= VALUE) then
      print *, "ERROR: target_parallel_variable_array (unexpected results)"
      stop 1
    end if
  end do
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: target_parallel_variable_array (too few threads)"
  else
    print *, "SUCCESS: target_parallel_variable_array"
  end if

  ! TODO: Test below segfaults when no explicit optimization option (-O0..-O3)
  ! is specified. That combination causes OpenMPOpt to run on unoptimized
  ! frontend-generated LLVM IR, whose shape triggers a bug that causes shared
  ! memory allocations for the allocatable array storage on device to be moved
  ! to scratch memory.
  ! CHECK: SUCCESS: target_parallel_allocatable_array
  seen1d = -1
  nthr = 0
  call target_parallel_allocatable_array(seen1d, nthr)
  do i = 1, nthr
    if (seen1d(i-1) /= VALUE) then
      print *, "ERROR: target_parallel_allocatable_array (unexpected results)"
      stop 1
    end if
  end do
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: target_parallel_allocatable_array (too few threads)"
  else
    print *, "SUCCESS: target_parallel_allocatable_array"
  end if

  ! CHECK: SUCCESS: target_teams_distribute_parallel_scalar
  seen2d = -1
  nthr = 0
  call target_teams_distribute_parallel_scalar(seen2d, nthr)
  do i = 1, NUM_TEAMS
    do j = 1, nthr
      if (seen2d(i-1, j-1) /= VALUE*2) then
        print *, "ERROR: target_teams_distribute_parallel_scalar (unexpected results)"
        stop 1
      end if
    end do
  end do
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: target_teams_distribute_parallel_scalar (too few threads)"
  else
    print *, "SUCCESS: target_teams_distribute_parallel_scalar"
  end if

  ! CHECK: SUCCESS: target_teams_distribute_parallel_array
  seen2d = -1
  nthr = 0
  call target_teams_distribute_parallel_array(seen2d, nthr)
  do i = 1, NUM_TEAMS
    do j = 1, nthr
      if (seen2d(i-1, j-1) /= VALUE*2) then
        print *, "ERROR: target_teams_distribute_parallel_array (unexpected results)"
        stop 1
      end if
    end do
  end do
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: target_teams_distribute_parallel_array (too few threads)"
  else
    print *, "SUCCESS: target_teams_distribute_parallel_array"
  end if

  ! CHECK: SUCCESS: target_teams_distribute_parallel_variable_array
  seen2d = -1
  nthr = 0
  call target_teams_distribute_parallel_variable_array(seen2d, nthr, 4)
  do i = 1, NUM_TEAMS
    do j = 1, nthr
      if (seen2d(i-1, j-1) /= VALUE*2) then
        print *, "ERROR: target_teams_distribute_parallel_variable_array (unexpected results)"
        stop 1
      end if
    end do
  end do
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: target_teams_distribute_parallel_variable_array (too few threads)"
  else
    print *, "SUCCESS: target_teams_distribute_parallel_variable_array"
  end if

  ! CHECK: SUCCESS: target_teams_distribute_parallel_allocatable_array
  seen2d = -1
  nthr = 0
  call target_teams_distribute_parallel_allocatable_array(seen2d, nthr)
  do i = 1, NUM_TEAMS
    do j = 1, nthr
      if (seen2d(i-1, j-1) /= VALUE*2) then
        print *, "ERROR: target_teams_distribute_parallel_allocatable_array (unexpected results)"
        stop 1
      end if
    end do
  end do
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: target_teams_distribute_parallel_allocatable_array (too few threads)"
  else
    print *, "SUCCESS: target_teams_distribute_parallel_allocatable_array"
  end if
end program main
