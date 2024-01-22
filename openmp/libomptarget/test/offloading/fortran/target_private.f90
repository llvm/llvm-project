! Basic offloading test with a target region
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
program target_update
    implicit none
    integer :: x(1)
    integer :: y(1)

    x(1) = 42

!$omp target private(x) map(tofrom: y)
    x(1) = 84
    y(1) = x(1)
!$omp end target

    print *, "x =", x(1)
    print *, "y =", y(1)

end program target_update
! CHECK: "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK: x = 42
! CHECK: y = 84
