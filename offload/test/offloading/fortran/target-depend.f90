! Offloading test checking the use of the depend clause on
! the target construct
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  integer :: a = 0
  call foo(5, a)
  print*, "======= FORTRAN Test passed! ======="
  print*, "foo(5) returned ", a, ", expected 8\n"
  !       stop 0
end program main
subroutine foo(N, r)
  integer, intent(in) :: N
  integer, intent(out) :: r
  integer :: z, i
  z = 1
  ! Spawn 3 threads
  !$omp parallel num_threads(3)

  ! Each thread redundantly updates z to N
  ! i.e. 5
  !$omp task depend(out: z) shared(z)
  do while (i < 32766)
     ! dumb loop to slow down the update of
     ! z
     i = i + 1
  end do
  z = N
  !$omp end task

  ! z is 5 now. Each thread then offloads
  ! increment of z by 1. So, z is incremented
  ! three times.
  !$omp target map(tofrom: z) depend(in: z)
  z = z + 1
  !$omp end target
  !$omp end parallel

  ! z is 8.
  r = z
end subroutine foo

!CHECK: ======= FORTRAN Test passed! =======
!CHECK: foo(5) returned 8 , expected 8
