! This test checks lowering of OpenMP unroll directive

! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines


program unroll_heuristic_do
  implicit none
  integer :: i
  print *, 'do'

  !$OMP UNROLL
  do i=7, 18, 3
    print '("i=", I0)', i
  end do
  !$OMP END UNROLL

  print *, 'done'
end program


! CHECK:      do
! CHECK-NEXT: i=7
! CHECK-NEXT: i=10
! CHECK-NEXT: i=13
! CHECK-NEXT: i=16
! CHECK-NEXT: done
