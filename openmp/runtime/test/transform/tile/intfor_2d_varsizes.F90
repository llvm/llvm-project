! This test checks lowering of OpenMP tile directive

! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

program tile_intfor_varsizes
  integer i

  call kernel(7,17,3,2)
  call kernel(7,17,3,3)

end program


subroutine kernel(lb, ub, step, ts)
  integer i, j, lb, ub, step, ts

  print *, 'do'

  !$OMP TILE SIZES(ts,ts)
  do i = lb, ub, step
    do j = 0, 2
      print '("i=", I0," j=", I0)', i, j
    end do
  end do
  !$OMP END TILE

  print *, 'done'

end subroutine

! CHECK:      do
! CHECK-NEXT: i=7 j=0
! CHECK-NEXT: i=7 j=1
! CHECK-NEXT: i=10 j=0
! CHECK-NEXT: i=10 j=1
! CHECK-NEXT: i=7 j=2
! CHECK-NEXT: i=10 j=2
! CHECK-NEXT: i=13 j=0
! CHECK-NEXT: i=13 j=1
! CHECK-NEXT: i=16 j=0
! CHECK-NEXT: i=16 j=1
! CHECK-NEXT: i=13 j=2
! CHECK-NEXT: i=16 j=2
! CHECK-NEXT: done

! CHECK:      do
! CHECK-NEXT: i=7 j=0
! CHECK-NEXT: i=7 j=1
! CHECK-NEXT: i=7 j=2
! CHECK-NEXT: i=10 j=0
! CHECK-NEXT: i=10 j=1
! CHECK-NEXT: i=10 j=2
! CHECK-NEXT: i=13 j=0
! CHECK-NEXT: i=13 j=1
! CHECK-NEXT: i=13 j=2
! CHECK-NEXT: i=16 j=0
! CHECK-NEXT: i=16 j=1
! CHECK-NEXT: i=16 j=2
! CHECK-NEXT: done
