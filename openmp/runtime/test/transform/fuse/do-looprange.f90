! RUN: %flang %flags %openmp_flags -fopenmp-version=60 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

program fuse_full
  implicit none
  integer i, j, k, u

  print *, 'do'

  !$OMP FUSE LOOPRANGE(2,2)
  do i=5, 25, 5
    print '("i=", I0)', i
  end do
  do j=10, 100, 10
    print '("j=", I0)', j
  end do
  do k=10, 0, -1
    print '("k=", I0)', k
  end do
  do u=5, 25, 5
    print '("u=", I0)', u
  end do
  !$OMP END FUSE

  print *, 'done'
end program

! CHECK: do
! CHECK-NEXT: i=5
! CHECK-NEXT: i=10
! CHECK-NEXT: i=15
! CHECK-NEXT: i=20
! CHECK-NEXT: i=25
! CHECK-NEXT: j=10
! CHECK-NEXT: k=10
! CHECK-NEXT: j=20
! CHECK-NEXT: k=9
! CHECK-NEXT: j=30
! CHECK-NEXT: k=8
! CHECK-NEXT: j=40
! CHECK-NEXT: k=7
! CHECK-NEXT: j=50
! CHECK-NEXT: k=6
! CHECK-NEXT: j=60
! CHECK-NEXT: k=5
! CHECK-NEXT: j=70
! CHECK-NEXT: k=4
! CHECK-NEXT: j=80
! CHECK-NEXT: k=3
! CHECK-NEXT: j=90
! CHECK-NEXT: k=2
! CHECK-NEXT: j=100
! CHECK-NEXT: k=1
! CHECK-NEXT: k=0
! CHECK-NEXT: u=5
! CHECK-NEXT: u=10
! CHECK-NEXT: u=15
! CHECK-NEXT: u=20
! CHECK-NEXT: u=25
! CHECK-NEXT: done
