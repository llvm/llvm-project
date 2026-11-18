! Every thread of a team takes the critical lock once, including threads that
! share a wavefront. A critical construct only excludes threads of the same
! contention group, so each team counts its own threads.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  use omp_lib
  implicit none
  integer, parameter :: nteams = 4
  integer :: counts(nteams), expected(nteams), t

  counts = 0
  expected = 0
  !$omp target teams num_teams(nteams) map(tofrom: counts, expected)
  !$omp parallel num_threads(256) private(t)
  t = omp_get_team_num() + 1
  !$omp critical
  counts(t) = counts(t) + 1
  !$omp end critical
  !$omp atomic
  expected(t) = expected(t) + 1
  !$omp end parallel
  !$omp end target teams

  if (all(counts == expected) .and. sum(counts) > 0) then
    print *, "PASS"
  else
    print *, "FAIL", counts, expected
  end if
end program main

! CHECK: PASS
