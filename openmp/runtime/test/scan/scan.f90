! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines
program inclusive_scan
 implicit none
 integer, parameter :: n = 100
 integer a(n), b(n)
 integer x, k, y, z

 ! initialization
 x = 0
 do k = 1, n
  a(k) = k
 end do

 ! a(k) is included in the computation of producing results in b(k)
 !$omp parallel do reduction(inscan, +: x)
 do k = 1, n
   x = x + a(k)
   !$omp scan inclusive(x)
   b(k) = x
 end do

 print *,'x =', x
 do k = 1, 10
  print *, 'b(', k, ') =', b(k)
 end do
end program
!CHECK: x = 5050
!CHECK: b( 1 ) = 1
!CHECK: b( 2 ) = 3
!CHECK: b( 3 ) = 6
!CHECK: b( 4 ) = 10
!CHECK: b( 5 ) = 15
!CHECK: b( 6 ) = 21
!CHECK: b( 7 ) = 28
!CHECK: b( 8 ) = 36
!CHECK: b( 9 ) = 45
!CHECK: b( 10 ) = 55
