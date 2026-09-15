! Offloading test for a collapsed imperfect loop nest whose bounds are
! evaluated on the host. Flattening the nest means the intervening statement
! would otherwise run once per (i,j) pair, so it is guarded on the inner
! induction variable; this checks it still runs exactly once per outer
! iteration on the device.

! REQUIRES: flang, gpu
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic

program main
   implicit none
   integer, parameter :: n = 10, m = 8
   integer :: i, j, errors
   integer :: after_inner(n, m), after_outer(n)
   integer :: before_inner(n, m), before_outer(n)

   after_inner = 0
   after_outer = 0
   before_inner = 0
   before_outer = 0

   ! Intervening code after the inner loop: guarded on j == its last value.
   !$omp target teams distribute parallel do collapse(2) map(tofrom: after_inner, after_outer)
   do i = 1, n
      do j = 1, m
         after_inner(i, j) = i + j
      end do
      after_outer(i) = after_outer(i) + 1
   end do

   ! Intervening code before the inner loop: guarded on j == its lower bound.
   !$omp target teams distribute parallel do collapse(2) map(tofrom: before_inner, before_outer)
   do i = 1, n
      before_outer(i) = before_outer(i) + 1
      do j = 1, m
         before_inner(i, j) = i * j
      end do
   end do

   errors = 0
   do i = 1, n
      ! A missing or misplaced guard makes this count m rather than 1.
      if (after_outer(i) /= 1) errors = errors + 1
      if (before_outer(i) /= 1) errors = errors + 1
      do j = 1, m
         if (after_inner(i, j) /= i + j) errors = errors + 1
         if (before_inner(i, j) /= i * j) errors = errors + 1
      end do
   end do

   print *, "number of errors: ", errors

end program main

! CHECK: PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK: number of errors: 0
