! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=1 %libomptarget-run-generic 2>&1 | %fcheck-generic
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=VALUE

module fp_info_mod
  type :: point
    integer :: x
    integer :: y
  end type
end module

program target_firstprivate_info
  use fp_info_mod
  implicit none
  integer :: factor, base, res
  integer :: arr(4)
  integer, allocatable :: pa(:)
  character(len=4) :: cstr
  type(point) :: p
  factor = 7; base = 5; res = 0
  arr = (/1, 2, 3, 4/)
  p%x = 3; p%y = 4
  allocate(pa(4)); pa = 1
  cstr = "abcd"

  ! 'factor' is explicitly privatized, 'base' is an implicit scalar capture.
  !$omp target map(tofrom: res) firstprivate(factor)
    res = factor + base
  !$omp end target
  print *, "res1 =", res

  ! array firstprivate is boxed for privatization; its descriptor maps must
  ! still report the source name.
  !$omp target map(tofrom: res) firstprivate(arr)
    res = arr(1) + arr(4)
  !$omp end target
  print *, "res2 =", res

  ! derived-type firstprivate
  !$omp target map(tofrom: res) firstprivate(p)
    res = p%x + p%y
  !$omp end target
  print *, "res3 =", res

  ! a fixed-size private array is not mapped, but a private allocatable maps its
  ! descriptor (copy left unallocated/unused here) and must report its name.
  !$omp target map(tofrom: res) private(pa)
    res = 99
  !$omp end target
  print *, "res4 =", res

  ! character firstprivate.
  !$omp target map(tofrom: res) firstprivate(cstr)
    res = ichar(cstr(1:1))
  !$omp end target
  print *, "res5 =", res
  deallocate(pa)
end program target_firstprivate_info

! CHECK: Entering OpenMP kernel {{.*}} with 3 arguments:
! CHECK-DAG: tofrom(res)[4]
! CHECK-DAG: firstprivate(base)[4] (implicit)
! CHECK-DAG: to(factor)[4]
! Array firstprivate is boxed; the descriptor maps still name the variable.
! CHECK: Entering OpenMP kernel {{.*}} with 5 arguments:
! CHECK-DAG: to(arr)[48]
! CHECK-DAG: attach(arr)[48]
! CHECK: Entering OpenMP kernel {{.*}} with 2 arguments:
! CHECK-DAG: tofrom(p)[8]
! A private allocatable and a character firstprivate are named too.
! CHECK: Entering OpenMP kernel {{.*}} with 5 arguments:
! CHECK-DAG: to(pa)[48]
! CHECK-DAG: attach(pa)[48]
! CHECK: Entering OpenMP kernel {{.*}} with 2 arguments:
! CHECK-DAG: to(cstr)[4]
! VALUE: res1 = 12
! VALUE: res2 = 5
! VALUE: res3 = 7
! VALUE: res4 = 99
! VALUE: res5 = 97
