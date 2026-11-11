! REQUIRES: flang
! RUN: %flang %flags %openmp_flags %s -o %t.exe
! RUN: %t.exe

! Fortran coverage of the 64-bit integer libomp entry points.
!
! The specific kmp_*_8 procedures are never named here. Every call uses the
! generic omp_*/kmp_* name with integer(8) actual arguments, so generic
! resolution in omp_lib is what selects the kmp_*_8 entry points. The 32-bit
! getters are used to verify the effects.

program kmp_integer_8_apis
  use omp_lib
  implicit none
  integer :: fails, i
  integer, parameter :: repetitions = 10

  fails = 0

  call test_disp_num_buffers()
  call test_stacksize()
  call test_blocktime()
  call test_library()

  do i = 1, repetitions
    call test_num_threads_and_team()
    call test_schedule()
    call test_max_active_levels()
    call test_teams()
    call test_default_device()
    call test_places()
    call test_affinity_mask()
  end do

  call test_pause_resource()

  if (fails /= 0) then
    print *, fails, ' check(s) failed'
    error stop 1
  end if
  print *, 'passed'

contains

  subroutine check(cond, msg)
    logical, intent(in) :: cond
    character(len=*), intent(in) :: msg
    if (.not. cond) then
      print *, 'FAIL: ', trim(msg)
      fails = fails + 1
    end if
  end subroutine check

  subroutine test_disp_num_buffers()
    call kmp_set_disp_num_buffers(8_8)
  end subroutine test_disp_num_buffers

  subroutine test_stacksize()
    integer(8) :: want
    integer(kmp_size_t_kind) :: got_s
    want = 8_8 * 1024_8 * 1024_8
    call kmp_set_stacksize(want)
    call check(kmp_get_stacksize() == int(want, kind=omp_integer_kind), &
         'kmp_set_stacksize(integer(8)) vs kmp_get_stacksize')
    got_s = kmp_get_stacksize_s()
    call check(got_s == int(want, kind=kmp_size_t_kind), &
         'kmp_set_stacksize(integer(8)) vs kmp_get_stacksize_s')
  end subroutine test_stacksize

  subroutine test_blocktime()
    call kmp_set_blocktime(200_8)
    call check(kmp_get_blocktime() == 200, 'kmp_set_blocktime(integer(8))')
  end subroutine test_blocktime

  subroutine test_library()
    ! enum library_type: none=0, serial=1, turnaround=2, throughput=3
    call kmp_set_library(3_8)
    call check(kmp_get_library() == 3, 'kmp_set_library(integer(8)) throughput')
    call kmp_set_library(2_8)
    call check(kmp_get_library() == 2, 'kmp_set_library(integer(8)) turnaround')
  end subroutine test_library

  subroutine test_num_threads_and_team()
    integer :: nthreads, nthreads_lib, team_err
    integer(8) :: level0, level1

    nthreads = 0
    nthreads_lib = -1
    team_err = 0
    level0 = 0_8
    level1 = 1_8

    call omp_set_dynamic(.false.)
    call omp_set_num_threads(2_8)
    call check(omp_get_max_threads() == 2, &
         'omp_set_num_threads(integer(8)) max_threads')

    call check(omp_get_ancestor_thread_num(level0) == 0, &
         'omp_get_ancestor_thread_num(0_8) serial')
    call check(omp_get_team_size(level0) == 1, &
         'omp_get_team_size(0_8) serial')

    !$omp parallel
    if (omp_get_ancestor_thread_num(level1) /= omp_get_thread_num() .or. &
        omp_get_team_size(level1) /= omp_get_num_threads() .or. &
        omp_get_ancestor_thread_num(level0) /= 0 .or. &
        omp_get_team_size(level0) /= 1) then
      !$omp atomic
      team_err = team_err + 1
    end if
    !$omp atomic
    nthreads = nthreads + 1
    !$omp single
    nthreads_lib = omp_get_num_threads()
    !$omp end single
    !$omp end parallel

    call check(team_err == 0, &
         'omp_get_ancestor_thread_num / omp_get_team_size with integer(8)')
    call check(nthreads == 2 .and. nthreads_lib == 2, 'parallel team size')

    call omp_set_num_threads(int(huge(0_4), kind=8) + 1000_8)
    call check(omp_get_max_threads() == huge(0_4), &
         'omp_set_num_threads(integer(8)) clamp to INT_MAX')
    call omp_set_num_threads(2_8)
  end subroutine test_num_threads_and_team

  subroutine test_schedule()
    integer(omp_sched_kind) :: kind
    integer(8) :: chunk

    kind = omp_sched_static
    chunk = 0_8
    call omp_set_schedule(omp_sched_dynamic, 7_8)
    call omp_get_schedule(kind, chunk)
    call check(kind == omp_sched_dynamic .and. chunk == 7_8, &
         'omp_set/get_schedule with integer(8) chunk, dynamic')

    call omp_set_schedule(omp_sched_guided, 1_8)
    call omp_get_schedule(kind, chunk)
    call check(kind == omp_sched_guided .and. chunk == 1_8, &
         'omp_set/get_schedule with integer(8) chunk, guided')
  end subroutine test_schedule

  subroutine test_max_active_levels()
    call omp_set_max_active_levels(1_8)
    call check(omp_get_max_active_levels() == 1, &
         'omp_set_max_active_levels(1_8)')
    call omp_set_max_active_levels(2_8)
    call check(omp_get_max_active_levels() >= 1, &
         'omp_set_max_active_levels(2_8)')
  end subroutine test_max_active_levels

  subroutine test_teams()
    call omp_set_num_teams(5_8)
    call check(omp_get_max_teams() == 5, 'omp_set_num_teams(5_8)')
    call omp_set_teams_thread_limit(7_8)
    call check(omp_get_teams_thread_limit() == 7, &
         'omp_set_teams_thread_limit(7_8)')
  end subroutine test_teams

  subroutine test_default_device()
    integer :: orig, after8, after32
    orig = omp_get_default_device()
    call omp_set_default_device(0_8)
    after8 = omp_get_default_device()
    call omp_set_default_device(0)
    after32 = omp_get_default_device()
    call check(after8 == after32, &
         'omp_set_default_device: integer(8) vs default integer')
    call omp_set_default_device(orig)
  end subroutine test_default_device

  subroutine test_places()
    integer :: nplaces, n32, n8, npart, i
    integer(omp_integer_kind), allocatable :: ids32(:), p32(:)
    integer(8), allocatable :: ids8(:), p8(:)
    integer(8) :: dummy(1)

    nplaces = omp_get_num_places()
    if (nplaces > 0) then
      n32 = omp_get_place_num_procs(0)
      n8 = omp_get_place_num_procs(0_8)
      call check(n32 == n8, 'omp_get_place_num_procs(0_8)')
      if (n32 > 0) then
        allocate(ids32(n32), ids8(n32))
        call omp_get_place_proc_ids(0, ids32)
        call omp_get_place_proc_ids(0_8, ids8)
        do i = 1, n32
          call check(ids32(i) == int(ids8(i), kind=omp_integer_kind), &
               'omp_get_place_proc_ids with integer(8) element')
        end do
        deallocate(ids32, ids8)
      end if
    else
      call check(omp_get_place_num_procs(0_8) == omp_get_place_num_procs(0), &
           'omp_get_place_num_procs(0_8) with no places')
    end if

    npart = omp_get_partition_num_places()
    if (npart > 0) then
      allocate(p32(npart), p8(npart))
      call omp_get_partition_place_nums(p32)
      call omp_get_partition_place_nums(p8)
      do i = 1, npart
        call check(p32(i) == int(p8(i), kind=omp_integer_kind), &
             'omp_get_partition_place_nums with integer(8) element')
      end do
      deallocate(p32, p8)
    else
      dummy(1) = -1_8
      call omp_get_partition_place_nums(dummy)
    end if
  end subroutine test_places

  subroutine test_affinity_mask()
    integer(kmp_affinity_mask_kind) :: mask
    integer :: max_proc, rc_set, rc_set32
    integer(8) :: proc0

    proc0 = 0_8
    call kmp_create_affinity_mask(mask)
    max_proc = kmp_get_affinity_max_proc()
    rc_set = kmp_set_affinity_mask_proc(proc0, mask)
    rc_set32 = kmp_set_affinity_mask_proc(0, mask)
    call check(rc_set == rc_set32, &
         'kmp_set_affinity_mask_proc: integer(8) vs default integer')
    if (rc_set == 0 .and. max_proc > 0) then
      call check(kmp_get_affinity_mask_proc(proc0, mask) == 1, &
           'kmp_get_affinity_mask_proc(integer(8)) after set')
      call check(kmp_unset_affinity_mask_proc(proc0, mask) == 0, &
           'kmp_unset_affinity_mask_proc(integer(8))')
      call check(kmp_get_affinity_mask_proc(proc0, mask) == 0, &
           'kmp_get_affinity_mask_proc(integer(8)) after unset')
    else
      rc_set = kmp_get_affinity_mask_proc(proc0, mask)
      rc_set = kmp_unset_affinity_mask_proc(proc0, mask)
    end if
    call kmp_destroy_affinity_mask(mask)
  end subroutine test_affinity_mask

  subroutine test_pause_resource()
    integer :: my_dev, nthreads
    integer(8) :: device8

    my_dev = omp_get_initial_device()
    device8 = int(my_dev, kind=8)
    nthreads = 0
    !$omp parallel
    !$omp single
    nthreads = omp_get_num_threads()
    !$omp end single
    !$omp end parallel
    call check(nthreads > 0, 'threads before pause')

    call check(omp_pause_resource(omp_pause_soft, device8) == 0, &
         'omp_pause_resource(integer(8) device) soft')
    nthreads = 0
    !$omp parallel
    !$omp single
    nthreads = omp_get_num_threads()
    !$omp end single
    !$omp end parallel
    call check(nthreads > 0, 'threads after pause soft')

    call check(omp_pause_resource(omp_pause_hard, device8) == 0, &
         'omp_pause_resource(integer(8) device) hard')
    nthreads = 0
    !$omp parallel
    !$omp single
    nthreads = omp_get_num_threads()
    !$omp end single
    !$omp end parallel
    call check(nthreads > 0, 'threads after pause hard')
  end subroutine test_pause_resource

end program kmp_integer_8_apis
