! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: env OMP_NUM_THREADS=4 %t.exe | FileCheck %s --match-full-lines

! A source-level `nowait` on a scan worksharing loop must only remove the final
! barrier after the scan (second) loop. The barrier at the end of the input
! (first) loop is required so that every thread finishes writing the temporary
! buffer before the masked prefix-sum reads it. Forcing multiple threads
! exercises the data race that occurred when `nowait` incorrectly removed that
! internal barrier.
program scan_nowait
 implicit none
 integer :: x, k
 integer :: b(8)

 b = -1
 x = 0
 !$omp parallel
 !$omp do reduction(inscan, +: x)
 do k = 1, 8
   x = x + 1
   !$omp scan inclusive(x)
   b(k) = x
 end do
 !$omp end do nowait
 !$omp end parallel

 do k = 1, 8
  print *, 'b(', k, ') =', b(k)
 end do
end program
!CHECK: b( 1 ) = 1
!CHECK: b( 2 ) = 2
!CHECK: b( 3 ) = 3
!CHECK: b( 4 ) = 4
!CHECK: b( 5 ) = 5
!CHECK: b( 6 ) = 6
!CHECK: b( 7 ) = 7
!CHECK: b( 8 ) = 8
