! REQUIRES: flang, amdgpu
! UNSUPPORTED: apu
! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMP_NUM_HIDDEN_HELPER_THREADS=0 %libomptarget-run-generic | %fcheck-generic

! Outstanding updates from the same call site must retain their own bounds.
program target_update_noncontiguous_nowait
  use omp_lib
  implicit none
  integer :: a(4,8,3), actual(4,8,3), i, b, k, release, worker_ready, ready

  do k = 1, 3
    do b = 1, 8
      do i = 1, 4
        a(i,b,k) = i + 10*b + 100*k
      end do
    end do
  end do
!$omp target enter data map(alloc:a)
!$omp target map(present,alloc:a)
  do k = 1, 3
    do b = 1, 8
      do i = 1, 4
        a(i,b,k) = -9999
      end do
    end do
  end do
!$omp end target

  release = 0
  worker_ready = 0
  call omp_set_dynamic(.false.)
!$omp parallel num_threads(2) shared(a,release,worker_ready) private(ready,b)
  if (omp_get_num_threads() /= 2) stop 2
  if (omp_get_thread_num() == 0) then
    ! With hidden helper threads disabled, keep the other worker outside the
    ! scheduler until all updates are queued. Each iteration has distinct bounds.
    do
!$omp atomic read
      ready = worker_ready
      if (ready /= 0) exit
    end do
    do b = 1, 8
!$omp target update to(a(:,b:b,:)) nowait
    end do
!$omp atomic write
    release = 1
!$omp taskwait
  else
!$omp atomic write
    worker_ready = 1
    do
!$omp atomic read
      ready = release
      if (ready /= 0) exit
    end do
  end if
!$omp end parallel

!$omp target map(present,alloc:a) map(from:actual)
  do k = 1, 3
    do b = 1, 8
      do i = 1, 4
        actual(i,b,k) = a(i,b,k)
      end do
    end do
  end do
!$omp end target
!$omp target exit data map(delete:a)
  if (any(actual /= a)) then
    print *, 'FAIL: incorrect elements:', count(actual /= a)
    stop 1
  end if
  print *, 'PASS'
end program

! CHECK: PASS
