! A `distribute` loop whose body contains a parallel region must keep its kernel
! in generic mode. __kmpc_distribute_static_loop_* spreads its iterations over
! the teams, not over the threads of a team: the runtime runs it with a team
! size of one and asserts the kernel is at parallel level 0, so exactly one
! thread per block may call it. If the kernel were SPMD-ized every thread would
! call it, each running the whole of its block's share of the body and entering
! the nested parallel region from level 0.
!
! Seeing the loop body does not change that, which is why this is worth a test:
! the body here is a definition the compiler can look inside, so the analysis
! knows every parallel region the kernel reaches. Acting on that by SPMD-izing
! the kernel faults with a memory access at a null address rather than giving a
! wrong answer quietly.
!
! Run at -O2 as well, since the mode is only decided once the optimizer runs.

! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-generic
! RUN: %libomptarget-run-generic | %fcheck-generic
! RUN: %libomptarget-compile-fortran-generic -O2
! RUN: %libomptarget-run-generic | %fcheck-generic

program main
  implicit none
  integer, parameter :: n = 10
  integer :: array(n, n), i, j, wrong

  array = 0

  !$omp target teams distribute map(tofrom: array)
  do i = 1, n
    !$omp parallel do
    do j = 1, n
      array(j, i) = i + j
    end do
  end do

  wrong = 0
  do i = 1, n
    do j = 1, n
      if (array(j, i) /= i + j) wrong = wrong + 1
    end do
  end do

  print *, "wrong elements:", wrong
end program main

! CHECK: wrong elements: 0
