! Test the Fortran interfaces of the OpenMP dynamic-groupprivate-information
! routines on the host and inside target regions, mirroring
! offload/test/offloading/dyn_groupprivate.cpp.
! REQUIRES: flang, gpu

! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61
! RUN: %libomptarget-run-generic | %fcheck-generic
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 -O3
! RUN: %libomptarget-run-generic | %fcheck-generic
! XFAIL: intelgpu

module dyn_gprivate_helpers
  use, intrinsic :: iso_c_binding, only : c_ptr, c_associated
  implicit none
contains
  ! C_ASSOCIATED(A, B) is false when A is null, so handle null pointers
  ! explicitly to compare two C pointers for equality.
  logical function same_ptr(a, b)
    !$omp declare target
    type(c_ptr), intent(in) :: a, b
    if (.not. c_associated(a)) then
      same_ptr = .not. c_associated(b)
    else
      same_ptr = c_associated(a, b)
    end if
  end function same_ptr
end module dyn_gprivate_helpers

program main
  use omp_lib
  use dyn_gprivate_helpers
  use, intrinsic :: iso_c_binding
  implicit none

  integer, parameter :: n = 512
  integer(c_int) :: res(n), buffer(n)
  integer(c_int), pointer :: dynbuf(:)
  integer :: nthreads, tid, wrapped, i, failed
  integer(c_size_t) :: max_size, exceeded_size

  failed = 0

  ! Verify the groupprivate buffer works as expected.
  !$omp target teams num_teams(1) thread_limit(n) &
  !$omp&   dyn_groupprivate(fallback(abort) : n * c_sizeof(0_c_int)) &
  !$omp&   map(from : res, nthreads) map(alloc : buffer)
  !$omp parallel private(dynbuf, tid, wrapped)
  call c_f_pointer(omp_get_dyn_gprivate_nofb_ptr(), dynbuf, [n])
  tid = omp_get_thread_num()
  if (tid == 0) nthreads = omp_get_num_threads()
  buffer(tid + 1) = 7
  dynbuf(tid + 1) = 3
  !$omp barrier
  wrapped = mod(tid + 37, nthreads)
  res(tid + 1) = buffer(wrapped + 1) + dynbuf(wrapped + 1)
  !$omp end parallel
  !$omp end target teams

  if (nthreads < n / 2 .or. nthreads > n) then
    print *, "Expected number of threads to be in [", n / 2, ":", n, &
             "], but got: ", nthreads
    stop 1
  end if

  do i = 1, nthreads
    if (res(i) /= 7 + 3) then
      print *, "res(", i, ") is ", res(i), ", expected ", 7 + 3
      failed = failed + 1
    end if
  end do

  ! Verify that the routines on the host return null and zero.
  if (c_associated(omp_get_dyn_gprivate_ptr())) failed = failed + 1
  if (c_associated(omp_get_dyn_gprivate_nofb_ptr())) failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= 0) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_null_mem_space) &
    failed = failed + 1

  max_size = omp_get_gprivate_limit(0, omp_access_cgroup)
  exceeded_size = max_size + 10

  ! Verify that the fallback(default_mem) modifier works.
  !$omp target dyn_groupprivate(fallback(default_mem) : exceeded_size) &
  !$omp&   map(tofrom : failed)
  if (.not. c_associated(omp_get_dyn_gprivate_ptr(0_c_size_t))) &
    failed = failed + 1
  if (c_associated(omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (same_ptr(omp_get_dyn_gprivate_ptr(0_c_size_t), &
               omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) failed = failed + 1
  if (omp_get_dyn_gprivate_size() == 0) failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= exceeded_size) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_default_mem_space) &
    failed = failed + 1
  !$omp end target

  ! Verify that the fallback(null) modifier works.
  !$omp target dyn_groupprivate(fallback(null) : exceeded_size) &
  !$omp&   map(tofrom : failed)
  if (c_associated(omp_get_dyn_gprivate_ptr(0_c_size_t))) failed = failed + 1
  if (c_associated(omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(0_c_size_t), &
                     omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= 0) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_null_mem_space) &
    failed = failed + 1
  !$omp end target

  ! Verify that the default modifier is fallback(default_mem).
  !$omp target dyn_groupprivate(exceeded_size) map(tofrom : failed)
  if (.not. c_associated(omp_get_dyn_gprivate_ptr(0_c_size_t))) &
    failed = failed + 1
  if (c_associated(omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (same_ptr(omp_get_dyn_gprivate_ptr(0_c_size_t), &
               omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) failed = failed + 1
  if (omp_get_dyn_gprivate_size() == 0) failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= exceeded_size) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_default_mem_space) &
    failed = failed + 1
  !$omp end target

  ! Verify that the fallback(abort) modifier works.
  !$omp target dyn_groupprivate(fallback(abort) : n) map(tofrom : failed)
  if (.not. c_associated(omp_get_dyn_gprivate_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. c_associated(omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(0_c_size_t), &
                     omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(5_c_size_t), &
                     omp_get_dyn_gprivate_nofb_ptr(5_c_size_t))) &
    failed = failed + 1
  if (omp_get_dyn_gprivate_size() == 0) failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= n) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_cgroup_mem_space) &
    failed = failed + 1
  !$omp end target

  ! Verify that omitted OPTIONAL arguments default to a zero offset and
  ! omp_access_cgroup.
  !$omp target dyn_groupprivate(fallback(abort) : n) map(tofrom : failed)
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(), &
                     omp_get_dyn_gprivate_ptr(0_c_size_t, omp_access_cgroup))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(access_group=omp_access_cgroup), &
                     omp_get_dyn_gprivate_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_nofb_ptr(), &
                     omp_get_dyn_gprivate_nofb_ptr(0_c_size_t, &
                                                   omp_access_cgroup))) &
    failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= &
      omp_get_dyn_gprivate_size(omp_access_cgroup)) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= &
      omp_get_dyn_gprivate_memspace(omp_access_cgroup)) failed = failed + 1
  !$omp end target

  ! Verify that the fallback(default_mem) does not trigger when not needed.
  !$omp target dyn_groupprivate(fallback(default_mem) : n) map(tofrom : failed)
  if (.not. c_associated(omp_get_dyn_gprivate_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. c_associated(omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(0_c_size_t), &
                     omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (omp_get_dyn_gprivate_size() == 0) failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= n) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_cgroup_mem_space) &
    failed = failed + 1
  !$omp end target

  ! Verify that the clause works when passing a zero size.
  !$omp target dyn_groupprivate(fallback(abort) : 0) map(tofrom : failed)
  if (c_associated(omp_get_dyn_gprivate_ptr(0_c_size_t))) failed = failed + 1
  if (c_associated(omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(0_c_size_t), &
                     omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= 0) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_null_mem_space) &
    failed = failed + 1
  !$omp end target

  ! Verify that the clause works when passing a zero size and
  ! fallback(default_mem).
  !$omp target dyn_groupprivate(fallback(default_mem) : 0) map(tofrom : failed)
  if (c_associated(omp_get_dyn_gprivate_ptr(0_c_size_t))) failed = failed + 1
  if (c_associated(omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(0_c_size_t), &
                     omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= 0) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_null_mem_space) &
    failed = failed + 1
  !$omp end target

  ! Verify that omitting the clause is the same as setting zero size.
  !$omp target map(tofrom : failed)
  if (c_associated(omp_get_dyn_gprivate_ptr(0_c_size_t))) failed = failed + 1
  if (c_associated(omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (.not. same_ptr(omp_get_dyn_gprivate_ptr(0_c_size_t), &
                     omp_get_dyn_gprivate_nofb_ptr(0_c_size_t))) &
    failed = failed + 1
  if (omp_get_dyn_gprivate_size() /= 0) failed = failed + 1
  if (omp_get_dyn_gprivate_memspace() /= omp_null_mem_space) &
    failed = failed + 1
  !$omp end target

  ! CHECK: PASS
  if (failed == 0) then
    print *, "PASS"
  else
    print *, "FAIL: ", failed
  end if
end program main