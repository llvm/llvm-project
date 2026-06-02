! Test declare target global replacement with USM reference pointer.
!
! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-generic -fopenmp-force-usm
! RUN: env LIBOMPTARGET_INFO=16 HSA_XNACK=1 %libomptarget-run-generic 2>&1 | %fcheck-generic

module m
    implicit none
    integer :: nx_vals
    !$omp declare target(nx_vals)
contains
    subroutine get_dims_noarg(kv)
        !$omp declare target
        integer, intent(out) :: kv
        kv = nx_vals
    end subroutine get_dims_noarg
end module m

program reproducer
    use m
    implicit none
    integer :: kv, kv_debug

    nx_vals = 6
    !$omp target enter data map(always, to: nx_vals)

    kv_debug = -1
    !$omp target map(tofrom: kv_debug)
    call get_dims_noarg(kv)
    kv_debug = kv
    !$omp end target

    print *, 'kv_debug after target (host)', kv_debug

    !$omp target exit data map(release: nx_vals)
end program reproducer

! CHECK: PluginInterface device {{[0-9]+}} info: Launching kernel
    ! CHECK: kv_debug after target (host) 6
