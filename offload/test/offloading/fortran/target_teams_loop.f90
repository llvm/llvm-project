! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
program target_teams_loop
    implicit none
    integer :: x(10), i

    !$omp target teams loop
    do i = 1, 10
      x(i) = i * 2
    end do

    print *, x
end program target_teams_loop

! CHECK: "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK: 2 4 6 8 10 12 14 16 18 20
