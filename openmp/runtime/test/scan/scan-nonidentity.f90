! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

! The reduction variable's non-identity incoming value must be included in the
! first partial result of an inclusive scan.
program scan_nonidentity
 implicit none
 integer :: x, k
 integer :: b(4)

 x = 10
 !$omp parallel do reduction(inscan, +: x)
 do k = 1, 4
   x = x + 1
   !$omp scan inclusive(x)
   b(k) = x
 end do

 print *, 'x =', x
 do k = 1, 4
   print *, 'b(', k, ') =', b(k)
 end do
end program
!CHECK: x = 14
!CHECK: b( 1 ) = 11
!CHECK: b( 2 ) = 12
!CHECK: b( 3 ) = 13
!CHECK: b( 4 ) = 14
