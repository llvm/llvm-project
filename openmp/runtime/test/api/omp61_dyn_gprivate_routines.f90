! RUN: %flang %flags %openmp_flags %s -o %t.exe
! RUN: %t.exe | FileCheck %s

! REQUIRES: linux

! Test the Fortran interfaces of the OpenMP dynamic-groupprivate-information
! routines and omp_get_gprivate_limit (TR15 Sec. 34.13).
!
! This exercises the interfaces, including the optional offset and access_group
! arguments, and checks the values reported by the host runtime. When no dynamic
! groupprivate block exists and no offload runtime is loaded, the routines report
! a null pointer, a size of zero, a null memory space and a zero limit.

program omp61_dyn_gprivate_routines
  use omp_lib
  use, intrinsic :: iso_c_binding, only : c_ptr, c_size_t, c_int, c_associated
  implicit none

  type(c_ptr) :: p
  integer(c_size_t) :: sz
  integer(omp_memspace_handle_kind) :: ms
  logical :: ok

  ok = .true.

  ! omp_get_dyn_gprivate_ptr: no block => null pointer. Cover default args,
  ! explicit offset, explicit access_group, and the offset-omitted-via-keyword
  ! form.
  p = omp_get_dyn_gprivate_ptr()
  if (c_associated(p)) ok = .false.
  p = omp_get_dyn_gprivate_ptr(0_c_size_t)
  if (c_associated(p)) ok = .false.
  p = omp_get_dyn_gprivate_ptr(0_c_size_t, omp_access_cgroup)
  if (c_associated(p)) ok = .false.
  p = omp_get_dyn_gprivate_ptr(access_group=omp_access_cgroup)
  if (c_associated(p)) ok = .false.

  ! omp_get_dyn_gprivate_nofb_ptr: no block => null pointer.
  p = omp_get_dyn_gprivate_nofb_ptr()
  if (c_associated(p)) ok = .false.
  p = omp_get_dyn_gprivate_nofb_ptr(0_c_size_t, omp_access_cgroup)
  if (c_associated(p)) ok = .false.

  ! omp_get_dyn_gprivate_size: no block => zero.
  sz = omp_get_dyn_gprivate_size()
  if (sz /= 0_c_size_t) ok = .false.
  sz = omp_get_dyn_gprivate_size(omp_access_cgroup)
  if (sz /= 0_c_size_t) ok = .false.

  ! omp_get_dyn_gprivate_memspace: no block => omp_null_mem_space.
  ms = omp_get_dyn_gprivate_memspace()
  if (ms /= omp_null_mem_space) ok = .false.
  ms = omp_get_dyn_gprivate_memspace(omp_access_cgroup)
  if (ms /= omp_null_mem_space) ok = .false.

  ! omp_get_gprivate_limit: the host reports no groupprivate limit.
  sz = omp_get_gprivate_limit(0_c_int)
  if (sz /= 0_c_size_t) ok = .false.
  sz = omp_get_gprivate_limit(0_c_int, omp_access_cgroup)
  if (sz /= 0_c_size_t) ok = .false.

  if (ok) then
    print *, "PASS"
  else
    print *, "FAIL"
  end if

! CHECK: PASS
end program omp61_dyn_gprivate_routines