! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

! Arbitrary loop bounds and a non-unit stride must index the scan buffer by the
! logical iteration number, not by the induction-variable value.
program scan_bounds
 implicit none
 integer, parameter :: n = 30
 integer :: a(n), b(n)
 integer :: x, k

 do k = 1, n
   a(k) = k
 end do

 ! Arbitrary bounds: iterate from 10 to 20.
 b = -1
 x = 0
 !$omp parallel do reduction(inscan, +: x)
 do k = 10, 20
   x = x + a(k)
   !$omp scan inclusive(x)
   b(k) = x
 end do
 print *, 'bounds x =', x
 do k = 10, 20
   print *, 'bb(', k, ') =', b(k)
 end do

 ! Non-unit stride: iterate from 1 to 10 step 2.
 b = -1
 x = 0
 !$omp parallel do reduction(inscan, +: x)
 do k = 1, 10, 2
   x = x + a(k)
   !$omp scan inclusive(x)
   b(k) = x
 end do
 print *, 'stride x =', x
 do k = 1, 10, 2
   print *, 'sb(', k, ') =', b(k)
 end do
end program
!CHECK: bounds x = 165
!CHECK: bb( 10 ) = 10
!CHECK: bb( 11 ) = 21
!CHECK: bb( 12 ) = 33
!CHECK: bb( 13 ) = 46
!CHECK: bb( 14 ) = 60
!CHECK: bb( 15 ) = 75
!CHECK: bb( 16 ) = 91
!CHECK: bb( 17 ) = 108
!CHECK: bb( 18 ) = 126
!CHECK: bb( 19 ) = 145
!CHECK: bb( 20 ) = 165
!CHECK: stride x = 25
!CHECK: sb( 1 ) = 1
!CHECK: sb( 3 ) = 4
!CHECK: sb( 5 ) = 9
!CHECK: sb( 7 ) = 16
!CHECK: sb( 9 ) = 25
