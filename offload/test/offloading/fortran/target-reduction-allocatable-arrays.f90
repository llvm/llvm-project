! Offloading test checking that reductions of allocatables and arrays work.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

module reduction_mod
  use omp_lib
  implicit none
  ! We set large enough launch bounds to ensure support is properly tested.
  ! Lower numbers of threads/teams can hide bugs when scratch memory is used
  ! where shared memory should be used instead.
  integer, parameter :: NUM_THREADS = 128
  integer, parameter :: MIN_THREADS = 64
  integer, parameter :: NUM_TEAMS = 64
  integer, parameter :: MIN_TEAMS = 2
  integer, parameter :: VALUE = 5
contains
  subroutine teams_scalar(result, nthr, nteams)
    integer, intent(out) :: result, nthr, nteams
    integer, allocatable :: a
    integer :: acc
    allocate(a)
    a = 0
    !$omp target teams map(from: nthr, nteams) thread_limit(NUM_THREADS) &
    !$omp& reduction(+: a) num_teams(NUM_TEAMS) private(acc)
      if (omp_get_team_num() == 0) nteams = omp_get_num_teams()
      a = a + VALUE
      acc = 0
      !$omp parallel reduction(+: acc) num_threads(NUM_THREADS)
        if (omp_get_team_num() == 0 .and. omp_get_thread_num() == 0) then
          nthr = omp_get_num_threads()
        end if
        acc = acc + a
      !$omp end parallel
      a = a + acc
    !$omp end target teams
    result = a
    deallocate(a)
  end subroutine

  subroutine teams_array(result, nthr, nteams)
    integer, intent(out) :: result, nthr, nteams
    integer :: a(4), acc
    a = 0
    !$omp target teams map(from: nthr, nteams) thread_limit(NUM_THREADS) &
    !$omp& reduction(+: a) num_teams(NUM_TEAMS) private(acc)
      if (omp_get_team_num() == 0) nteams = omp_get_num_teams()
      a(3) = a(3) + VALUE
      acc = 0
      !$omp parallel reduction(+: acc) num_threads(NUM_THREADS)
        if (omp_get_team_num() == 0 .and. omp_get_thread_num() == 0) then
          nthr = omp_get_num_threads()
        end if
        acc = acc + a(3)
      !$omp end parallel
      a(3) = a(3) + acc
    !$omp end target teams
    result = a(3)
  end subroutine

  ! TODO: Reduction of dynamically-shaped arrays on the GPU.
  ! Uncomment this and uses below to test the feature once implemented.
  ! subroutine teams_variable_array(result, nthr, nteams, sz)
  !   integer, intent(out) :: result, nthr, nteams
  !   integer, intent(in) :: sz
  !   integer :: a(sz), acc
  !   a = 0
  !   !$omp target teams map(from: nthr, nteams) thread_limit(NUM_THREADS) &
  !   !$omp& reduction(+: a) num_teams(NUM_TEAMS) private(acc)
  !     if (omp_get_team_num() == 0) nteams = omp_get_num_teams()
  !     a(3) = a(3) + VALUE
  !     acc = 0
  !     !$omp parallel reduction(+: acc) num_threads(NUM_THREADS)
  !       if (omp_get_team_num() == 0 .and. omp_get_thread_num() == 0) then
  !         nthr = omp_get_num_threads()
  !       end if
  !       acc = acc + a(3)
  !     !$omp end parallel
  !     a(3) = a(3) + acc
  !   !$omp end target teams
  !   result = a(3)
  ! end subroutine

  ! TODO: Reduction of dynamically-shaped arrays on the GPU.
  ! Uncomment this and uses below to test the feature once implemented.
  ! subroutine teams_allocatable_array(result, nthr, nteams)
  !   integer, intent(out) :: result, nthr, nteams
  !   integer, allocatable :: a(:)
  !   integer :: acc
  !   allocate(a(4))
  !   a = 0
  !   !$omp target teams map(from: nthr, nteams) thread_limit(NUM_THREADS) &
  !   !$omp& reduction(+: a) num_teams(NUM_TEAMS) private(acc)
  !     if (omp_get_team_num() == 0) nteams = omp_get_num_teams()
  !     a(3) = a(3) + VALUE
  !     acc = 0
  !     !$omp parallel reduction(+: acc) num_threads(NUM_THREADS)
  !       if (omp_get_team_num() == 0 .and. omp_get_thread_num() == 0) then
  !         nthr = omp_get_num_threads()
  !       end if
  !       acc = acc + a(3)
  !     !$omp end parallel
  !     a(3) = a(3) + acc
  !   !$omp end target teams
  !   result = a(3)
  !   deallocate(a)
  ! end subroutine

  subroutine teams_distribute_scalar(result, nthr, nteams)
    integer, intent(out) :: result, nthr, nteams
    integer, allocatable :: a
    integer :: acc, i
    allocate(a)
    a = 0
    !$omp target teams map(from: nthr, nteams) thread_limit(NUM_THREADS) &
    !$omp& reduction(+: a) num_teams(NUM_TEAMS)
      if (omp_get_team_num() == 0) nteams = omp_get_num_teams()
      !$omp distribute private(acc) dist_schedule(static, 1)
      do i = 1, nteams
        a = a + VALUE
        acc = 0
        !$omp parallel reduction(+: acc) num_threads(NUM_THREADS)
          if (omp_get_team_num() == 0 .and. omp_get_thread_num() == 0) then
            nthr = omp_get_num_threads()
          end if
          acc = acc + a
        !$omp end parallel
        a = a + acc
      end do
    !$omp end target teams
    result = a
    deallocate(a)
  end subroutine

  subroutine teams_distribute_array(result, nthr, nteams)
    integer, intent(out) :: result, nthr, nteams
    integer :: a(4), acc, i
    a = 0
    !$omp target teams map(from: nthr, nteams) thread_limit(NUM_THREADS) &
    !$omp& reduction(+: a) num_teams(NUM_TEAMS)
      if (omp_get_team_num() == 0) nteams = omp_get_num_teams()
      !$omp distribute private(acc) dist_schedule(static, 1)
      do i = 1, nteams
        a(3) = a(3) + VALUE
        acc = 0
        !$omp parallel reduction(+: acc) num_threads(NUM_THREADS)
          if (omp_get_team_num() == 0 .and. omp_get_thread_num() == 0) then
            nthr = omp_get_num_threads()
          end if
          acc = acc + a(3)
        !$omp end parallel
        a(3) = a(3) + acc
      end do
    !$omp end target teams
    result = a(3)
  end subroutine

  ! TODO: Reduction of dynamically-shaped arrays on the GPU.
  ! Uncomment this and uses below to test the feature once implemented.
  ! subroutine teams_distribute_variable_array(result, nthr, nteams, sz)
  !   integer, intent(out) :: result, nthr, nteams
  !   integer, intent(in) :: sz
  !   integer :: a(sz), acc, i
  !   a = 0
  !   !$omp target teams map(from: nthr, nteams) thread_limit(NUM_THREADS) &
  !   !$omp& reduction(+: a) num_teams(NUM_TEAMS)
  !     if (omp_get_team_num() == 0) nteams = omp_get_num_teams()
  !     !$omp distribute private(acc) dist_schedule(static, 1)
  !     do i = 1, nteams
  !       a(3) = a(3) + VALUE
  !       acc = 0
  !       !$omp parallel reduction(+: acc) num_threads(NUM_THREADS)
  !         if (omp_get_team_num() == 0 .and. omp_get_thread_num() == 0) then
  !           nthr = omp_get_num_threads()
  !         end if
  !         acc = acc + a(3)
  !       !$omp end parallel
  !       a(3) = a(3) + acc
  !     end do
  !   !$omp end target teams
  !   result = a(3)
  ! end subroutine

  ! TODO: Reduction of dynamically-shaped arrays on the GPU.
  ! Uncomment this and uses below to test the feature once implemented.
  ! subroutine teams_distribute_allocatable_array(result, nthr, nteams)
  !   integer, intent(out) :: result, nthr, nteams
  !   integer, allocatable :: a(:)
  !   integer :: acc, i
  !   allocate(a(4))
  !   a = 0
  !   !$omp target teams map(from: nthr, nteams) thread_limit(NUM_THREADS) &
  !   !$omp& reduction(+: a) num_teams(NUM_TEAMS)
  !     if (omp_get_team_num() == 0) nteams = omp_get_num_teams()
  !     !$omp distribute private(acc) dist_schedule(static, 1)
  !     do i = 1, nteams
  !       a(3) = a(3) + VALUE
  !       acc = 0
  !       !$omp parallel reduction(+: acc) num_threads(NUM_THREADS)
  !         if (omp_get_team_num() == 0 .and. omp_get_thread_num() == 0) then
  !           nthr = omp_get_num_threads()
  !         end if
  !         acc = acc + a(3)
  !       !$omp end parallel
  !       a(3) = a(3) + acc
  !     end do
  !   !$omp end target teams
  !   result = a(3)
  !   deallocate(a)
  ! end subroutine
end module

program main
  use reduction_mod
  implicit none
  integer :: result, nthr, nteams, expected

  ! CHECK: SUCCESS: teams_scalar
  result = -1
  nthr = 0
  nteams = 0
  call teams_scalar(result, nthr, nteams)
  expected = nteams * VALUE * (1 + nthr)
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: teams_scalar (too few threads)"
  else if (nteams <= MIN_TEAMS) then
    print *, "INCONCLUSIVE: teams_scalar (too few teams)"
  else if (result /= expected) then
    print *, "ERROR: teams_scalar (unexpected result:", result, "vs", expected , ")"
  else
    print *, "SUCCESS: teams_scalar"
  end if

  ! CHECK: SUCCESS: teams_array
  result = -1
  nthr = 0
  nteams = 0
  call teams_array(result, nthr, nteams)
  expected = nteams * VALUE * (1 + nthr)
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: teams_array (too few threads)"
  else if (nteams <= MIN_TEAMS) then
    print *, "INCONCLUSIVE: teams_array (too few teams)"
  else if (result /= expected) then
    print *, "ERROR: teams_array (unexpected result:", result, "vs", expected , ")"
  else
    print *, "SUCCESS: teams_array"
  end if

  ! ! CHECK: SUCCESS: teams_variable_array
  ! result = -1
  ! nthr = 0
  ! nteams = 0
  ! call teams_variable_array(result, nthr, nteams, 4)
  ! expected = nteams * VALUE * (1 + nthr)
  ! if (nthr < MIN_THREADS) then
  !   print *, "INCONCLUSIVE: teams_variable_array (too few threads)"
  ! else if (nteams <= MIN_TEAMS) then
  !   print *, "INCONCLUSIVE: teams_variable_array (too few teams)"
  ! else if (result /= expected) then
  !   print *, "ERROR: teams_variable_array (unexpected result:", result, "vs", expected , ")"
  ! else
  !   print *, "SUCCESS: teams_variable_array"
  ! end if

  ! ! CHECK: SUCCESS: teams_allocatable_array
  ! result = -1
  ! nthr = 0
  ! nteams = 0
  ! call teams_allocatable_array(result, nthr, nteams)
  ! expected = nteams * VALUE * (1 + nthr)
  ! if (nthr < MIN_THREADS) then
  !   print *, "INCONCLUSIVE: teams_allocatable_array (too few threads)"
  ! else if (nteams <= MIN_TEAMS) then
  !   print *, "INCONCLUSIVE: teams_allocatable_array (too few teams)"
  ! else if (result /= expected) then
  !   print *, "ERROR: teams_allocatable_array (unexpected result:", result, "vs", expected , ")"
  ! else
  !   print *, "SUCCESS: teams_allocatable_array"
  ! end if

  ! CHECK: SUCCESS: teams_distribute_scalar
  result = -1
  nthr = 0
  nteams = 0
  call teams_distribute_scalar(result, nthr, nteams)
  expected = nteams * VALUE * (1 + nthr)
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: teams_distribute_scalar (too few threads)"
  else if (nteams <= MIN_TEAMS) then
    print *, "INCONCLUSIVE: teams_distribute_scalar (too few teams)"
  else if (result /= expected) then
    print *, "ERROR: teams_distribute_scalar (unexpected result:", result, "vs", expected , ")"
  else
    print *, "SUCCESS: teams_distribute_scalar"
  end if

  ! CHECK: SUCCESS: teams_distribute_array
  result = -1
  nthr = 0
  nteams = 0
  call teams_distribute_array(result, nthr, nteams)
  expected = nteams * VALUE * (1 + nthr)
  if (nthr < MIN_THREADS) then
    print *, "INCONCLUSIVE: teams_distribute_array (too few threads)"
  else if (nteams <= MIN_TEAMS) then
    print *, "INCONCLUSIVE: teams_distribute_array (too few teams)"
  else if (result /= expected) then
    print *, "ERROR: teams_distribute_array (unexpected result:", result, "vs", expected , ")"
  else
    print *, "SUCCESS: teams_distribute_array"
  end if

  ! ! CHECK: SUCCESS: teams_distribute_variable_array
  ! result = -1
  ! nthr = 0
  ! nteams = 0
  ! call teams_distribute_variable_array(result, nthr, nteams, 4)
  ! expected = nteams * VALUE * (1 + nthr)
  ! if (nthr < MIN_THREADS) then
  !   print *, "INCONCLUSIVE: teams_distribute_variable_array (too few threads)"
  ! else if (nteams <= MIN_TEAMS) then
  !   print *, "INCONCLUSIVE: teams_distribute_variable_array (too few teams)"
  ! else if (result /= expected) then
  !   print *, "ERROR: teams_distribute_variable_array (unexpected result:", result, "vs", expected , ")"
  ! else
  !   print *, "SUCCESS: teams_distribute_variable_array"
  ! end if

  ! ! CHECK: SUCCESS: teams_distribute_allocatable_array
  ! result = -1
  ! nthr = 0
  ! nteams = 0
  ! call teams_distribute_allocatable_array(result, nthr, nteams)
  ! expected = nteams * VALUE * (1 + nthr)
  ! if (nthr < MIN_THREADS) then
  !   print *, "INCONCLUSIVE: teams_distribute_allocatable_array (too few threads)"
  ! else if (nteams <= MIN_TEAMS) then
  !   print *, "INCONCLUSIVE: teams_distribute_allocatable_array (too few teams)"
  ! else if (result /= expected) then
  !   print *, "ERROR: teams_distribute_allocatable_array (unexpected result:", result, "vs", expected , ")"
  ! else
  !   print *, "SUCCESS: teams_distribute_allocatable_array"
  ! end if
end program main
