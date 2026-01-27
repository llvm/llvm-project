! RUN: %flang %flags %openmp_flags -fopenmp-version=60 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

program interchange
  implicit none
  integer :: i, j, k

  print *, 'do'

  !$omp interchange
  do i = 1, 5
    do j = 1, 2
      print '("i=", I0, " j=", I1)', i, j
    end do
  end do

  print *, 'done'

end program

! CHECK:      do
! CHECK-NEXT: i=1 j=1
! CHECK-NEXT: i=2 j=1
! CHECK-NEXT: i=3 j=1
! CHECK-NEXT: i=4 j=1
! CHECK-NEXT: i=5 j=1
! CHECK-NEXT: i=1 j=2
! CHECK-NEXT: i=2 j=2
! CHECK-NEXT: i=3 j=2
! CHECK-NEXT: i=4 j=2
! CHECK-NEXT: i=5 j=2
! CHECK-NEXT: done
