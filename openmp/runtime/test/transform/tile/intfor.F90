! This test checks lowering of the OpenMP tile directive
! It is done 3 times corresponding to every possible fraction of the last
! iteration before passing beyond UB.

! RUN: %flang %flags %openmp_flags -fopenmp-version=51 -DUB=16 %s -o %t-ub16.exe
! RUN: %flang %flags %openmp_flags -fopenmp-version=51 -DUB=17 %s -o %t-ub17.exe
! RUN: %flang %flags %openmp_flags -fopenmp-version=51 -DUB=18 %s -o %t-ub18.exe
! RUN: %t-ub16.exe | FileCheck %s --match-full-lines
! RUN: %t-ub17.exe | FileCheck %s --match-full-lines
! RUN: %t-ub18.exe | FileCheck %s --match-full-lines

program tile_intfor_1d
  implicit none
  integer i
  print *, 'do'

  !$OMP TILE SIZES(2)
  do i=7, UB, 3
    print '("i=", I0)', i
  end do
  !$OMP END TILE

  print *, 'done'
end program


! CHECK:      do
! CHECK-NEXT: i=7
! CHECK-NEXT: i=10
! CHECK-NEXT: i=13
! CHECK-NEXT: i=16
! CHECK-NEXT: done
