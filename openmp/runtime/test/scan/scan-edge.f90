! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

! Zero-trip loops must not crash and must leave the reduction variable
! unchanged, and runtime (non-constant) trip counts must work correctly.
module scan_edge_mod
contains
 subroutine run_scan(n, a, b)
   implicit none
   integer :: n, k
   integer :: a(n), b(n)
   integer :: x
   x = 0
   !$omp parallel do reduction(inscan, +: x)
   do k = 1, n
     x = x + a(k)
     !$omp scan inclusive(x)
     b(k) = x
   end do
 end subroutine
end module

program scan_edge
 use scan_edge_mod
 implicit none
 integer :: a(6), b(6), k
 integer :: z, zk
 integer :: zb(1)

 ! Runtime bounds: the trip count of run_scan is not a compile-time constant.
 do k = 1, 6
   a(k) = k
 end do
 b = -1
 call run_scan(6, a, b)
 do k = 1, 6
   print *, 'rb(', k, ') =', b(k)
 end do

 ! Runtime zero-trip: n = 0 must not crash.
 call run_scan(0, a, b)
 print *, 'runtime zero-trip ok'

 ! Constant zero-trip with a non-identity start: z must be unchanged.
 z = 42
 !$omp parallel do reduction(inscan, +: z)
 do zk = 1, 0
   z = z + 1
   !$omp scan inclusive(z)
   zb(zk) = z
 end do
 print *, 'z =', z
end program
!CHECK: rb( 1 ) = 1
!CHECK: rb( 2 ) = 3
!CHECK: rb( 3 ) = 6
!CHECK: rb( 4 ) = 10
!CHECK: rb( 5 ) = 15
!CHECK: rb( 6 ) = 21
!CHECK: runtime zero-trip ok
!CHECK: z = 42
