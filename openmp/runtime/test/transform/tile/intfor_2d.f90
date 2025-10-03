! This test checks lowering of OpenMP tile directive

! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines


program tile_intfor_2d
  integer i, j
  print *, 'do'

  !$OMP TILE SIZES(2,3)
  do i = 7, 16, 3
    do j = 0, 4
      print '("i=", I0," j=", I0)', i, j
    end do
  end do
  !$OMP END TILE

  print *, 'done'
end program


! CHECK:      do

! complete tile
! CHECK-NEXT: i=7 j=0
! CHECK-NEXT: i=7 j=1
! CHECK-NEXT: i=7 j=2
! CHECK-NEXT: i=10 j=0
! CHECK-NEXT: i=10 j=1
! CHECK-NEXT: i=10 j=2

! partial tile
! CHECK-NEXT: i=7 j=3
! CHECK-NEXT: i=7 j=4
! CHECK-NEXT: i=10 j=3
! CHECK-NEXT: i=10 j=4

! complete tile
! CHECK-NEXT: i=13 j=0
! CHECK-NEXT: i=13 j=1
! CHECK-NEXT: i=13 j=2
! CHECK-NEXT: i=16 j=0
! CHECK-NEXT: i=16 j=1
! CHECK-NEXT: i=16 j=2

! partial tile
! CHECK-NEXT: i=13 j=3
! CHECK-NEXT: i=13 j=4
! CHECK-NEXT: i=16 j=3
! CHECK-NEXT: i=16 j=4

! CHECK-NEXT: done
