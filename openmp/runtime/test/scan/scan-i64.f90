! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

! An `integer(kind=8)` (i64) loop induction variable must work with an inscan
! reduction. The scan temporary buffer is indexed using the loop's own index
! type, so an i64 induction variable must not trip the i32/i64 type mismatch in
! the scan buffer allocation or the prefix-sum computation.
program scan_i64
 implicit none
 integer(kind=8) :: i, n
 integer :: k
 integer :: x
 integer :: b(8)

 n = 8
 x = 0
 !$omp parallel do reduction(inscan, +: x)
 do i = 1_8, n
   x = x + 1
   !$omp scan inclusive(x)
   b(i) = x
 end do

 print *, 'x =', x
 do k = 1, 8
  print *, 'b(', k, ') =', b(k)
 end do
end program
!CHECK: x = 8
!CHECK: b( 1 ) = 1
!CHECK: b( 2 ) = 2
!CHECK: b( 3 ) = 3
!CHECK: b( 4 ) = 4
!CHECK: b( 5 ) = 5
!CHECK: b( 6 ) = 6
!CHECK: b( 7 ) = 7
!CHECK: b( 8 ) = 8
