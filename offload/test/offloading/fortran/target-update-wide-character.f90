! REQUIRES: flang, amdgpu
! UNSUPPORTED: apu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

! Assumed-length character bounds count code units, not bytes. Check every
! code unit on the device after TO and on the host after FROM.
program target_update_wide_character
  implicit none
  character(kind=2,len=8) :: a2
  character(kind=4,len=8) :: a4
  call check_kind2(a2)
  call check_kind4(a4)
  call absent_kind4()
  print *, 'PASS'
contains
  subroutine check_kind2(a)
    character(kind=2,len=*) :: a
    character(kind=2,len=8), parameter :: to_value = 2_'A B C   '
    character(kind=2,len=8), parameter :: from_value = 2_'X Y Z   '
    integer :: actual(8), i
!$omp target enter data map(alloc:a)
!$omp target map(present,alloc:a)
    a = 2_'device  '
!$omp end target
    a = to_value
!$omp target update to(a)
!$omp target map(present,alloc:a) map(from:actual)
    do i = 1, 8
      actual(i) = iachar(a(i:i))
    end do
    a = from_value
!$omp end target
    do i = 1, 8
      if (actual(i) /= iachar(to_value(i:i))) stop 1
    end do
    a = 2_'hostdata'
!$omp target update from(a)
    if (a /= from_value) stop 2
!$omp target exit data map(delete:a)
  end subroutine

  subroutine check_kind4(a)
    character(kind=4,len=*) :: a
    character(kind=4,len=8), parameter :: to_value = 4_'A B C   '
    character(kind=4,len=8), parameter :: from_value = 4_'X Y Z   '
    integer :: actual(8), i
!$omp target enter data map(alloc:a)
!$omp target map(present,alloc:a)
    a = 4_'device  '
!$omp end target
    a = to_value
!$omp target update to(a)
!$omp target map(present,alloc:a) map(from:actual)
    do i = 1, 8
      actual(i) = iachar(a(i:i))
    end do
    a = from_value
!$omp end target
    do i = 1, 8
      if (actual(i) /= iachar(to_value(i:i))) stop 3
    end do
    a = 4_'hostdata'
!$omp target update from(a)
    if (a /= from_value) stop 4
!$omp target exit data map(delete:a)
  end subroutine

  subroutine absent_kind4(a)
    character(kind=4,len=*), optional :: a
!$omp target update to(present:a)
!$omp target update from(present:a)
  end subroutine
end program

! CHECK: PASS
