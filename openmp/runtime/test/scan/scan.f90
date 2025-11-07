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
end program
!CHECK: x = 5050
