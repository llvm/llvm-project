! REQUIRES: flang, flang-rt, libc, amdgpu
! UNSUPPORTED: apu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

! A section of the second dimension has gaps between the (k, s) slices.
! Check both the values transferred and the values outside the section.
program target_update_noncontiguous
  implicit none
  type :: waves
    integer, pointer :: a(:,:,:,:)
  end type
  type(waves), target :: w
  integer :: actual(0:2, 5, -1:0, 2:3)
  integer :: values(0:2, 5, -1:0, 2:3)
  integer :: expected(0:2, 5, -1:0, 2:3)
  integer :: plain(-2:0, 5, 2, 2), plain_actual(-2:0, 5, 2, 2)
  integer :: i, b, k, s, nb
  integer, parameter :: host_sentinel = 7777, device_sentinel = -9999

  nb = 2
  allocate(w%a(0:2, 5, -1:0, 2:3))
  do s = 2, 3
    do k = -1, 0
      do b = 1, 5
        do i = 0, 2
          values(i,b,k,s) = i + 10*b + 100*k + 1000*s
        end do
      end do
    end do
  end do
  w%a = values
!$omp target enter data map(to:w)
!$omp target enter data map(alloc:w%a)
!$omp target map(present,to:w)
  w%a = device_sentinel
!$omp end target

!$omp target update to(w%a(:,nb+1:,:,:))
  ! Read through a kernel: a whole-array update would hide the original bug.
!$omp target map(present,to:w) map(from:actual)
  do s = 2, 3
    do k = -1, 0
      do b = 1, 5
        do i = 0, 2
          actual(i,b,k,s) = w%a(i,b,k,s)
          w%a(i,b,k,s) = -(i + 10*b + 100*k + 1000*s)
        end do
      end do
    end do
  end do
!$omp end target
  expected = device_sentinel
  expected(:,nb+1:,:,:) = values(:,nb+1:,:,:)
  if (any(actual /= expected)) stop 1

  w%a = host_sentinel
!$omp target update from(w%a(:,nb+1:,:,:))
  expected = host_sentinel
  expected(:,nb+1:,:,:) = -values(:,nb+1:,:,:)
  if (any(w%a /= expected)) stop 2
!$omp target exit data map(delete:w%a)
!$omp target exit data map(delete:w)
  deallocate(w%a)

  ! Explicit-shape arrays also require gaps between the outer slices. Keep
  ! a nondefault lower bound in the innermost dimension.
  plain = 42
!$omp target enter data map(alloc:plain)
!$omp target map(present,to:plain)
  plain = device_sentinel
!$omp end target
!$omp target update to(plain(:,nb+1:,:,:))
!$omp target map(present,to:plain) map(from:plain_actual)
  plain_actual = plain
  plain = -42
!$omp end target
  do s = 1, 2
    do k = 1, 2
      do b = 1, 5
        do i = -2, 0
          if (b > nb) then
            if (plain_actual(i,b,k,s) /= 42) stop 3
          else
            if (plain_actual(i,b,k,s) /= device_sentinel) stop 4
          end if
        end do
      end do
    end do
  end do
  plain = host_sentinel
!$omp target update from(plain(:,nb+1:,:,:))
  do s = 1, 2
    do k = 1, 2
      do b = 1, 5
        do i = -2, 0
          if (b > nb) then
            if (plain(i,b,k,s) /= -42) stop 5
          else
            if (plain(i,b,k,s) /= host_sentinel) stop 6
          end if
        end do
      end do
    end do
  end do
  ! A scalar subscript selects one element but retains the physical first
  ! dimension extent when calculating the stride of subsequent dimensions.
  plain = 84
!$omp target map(present,to:plain)
  plain = device_sentinel
!$omp end target
!$omp target update to(plain(0,nb+1:,:,:))
!$omp target map(present,to:plain) map(from:plain_actual)
  plain_actual = plain
  plain = -84
!$omp end target
  if (any(plain_actual(0,nb+1:,:,:) /= 84)) stop 7
  plain_actual(0,nb+1:,:,:) = device_sentinel
  if (any(plain_actual /= device_sentinel)) stop 8

  plain = host_sentinel
!$omp target update from(plain(0,nb+1:,:,:))
  if (any(plain(0,nb+1:,:,:) /= -84)) stop 9
  plain(0,nb+1:,:,:) = host_sentinel
  if (any(plain /= host_sentinel)) stop 10
!$omp target exit data map(delete:plain)
  call update_absent()
  print *, "PASS"
contains
  subroutine update_absent(a)
    integer, optional :: a(4,5)
    ! An absent explicit-shape dummy has a null raw base rather than a null
    ! descriptor. PRESENT must not request a mapping for this empty update.
!$omp target update to(present:a(2:3,:))
!$omp target update from(present:a(2:3,:))
  end subroutine
end program

! CHECK: PASS
