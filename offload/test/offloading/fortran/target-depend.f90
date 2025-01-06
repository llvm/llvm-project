! Offloading test checking the use of the depend clause on the target construct
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-unknown-linux-gnu
! UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  implicit none
  integer :: a = 0
  INTERFACE
     FUNCTION omp_get_device_num() BIND(C)
       USE, INTRINSIC :: iso_c_binding, ONLY: C_INT
       integer :: omp_get_device_num
     END FUNCTION omp_get_device_num
  END INTERFACE

  call foo(5, a)
  print*, "======= FORTRAN Test passed! ======="
  print*, "foo(5) returned ", a, ", expected 6\n"

  !       stop 0
  contains
    subroutine foo(N, r)
      integer, intent(in) :: N
      integer, intent(out) :: r
      integer :: z, i, accumulator
      z = 1
      accumulator = 0
      ! Spawn 3 threads
      !$omp parallel num_threads(3)

      ! A single thread will then create two tasks - one is the 'producer' and
      ! potentially slower task that updates 'z' to 'N'. The second is an
      ! offloaded target task that increments 'z'. If the depend clauses work
      ! properly, the target task should wait for the 'producer' task to
      ! complete before incrementing 'z'. We use 'omp single' here because the
      ! depend clause establishes dependencies between sibling tasks only.
      ! This is the easiest way of creating two sibling tasks.
      !$omp single
      !$omp task depend(out: z) shared(z)
      do i=1, 32766
         ! dumb loop nest to slow down the update of 'z'.
         ! Adding a function call slows down the producer to the point
         ! that removing the depend clause from the target construct below
         ! frequently results in the wrong answer.
         accumulator = accumulator + omp_get_device_num()
      end do
      z = N
      !$omp end task

      ! z is 5 now. Increment z to 6.
      !$omp target map(tofrom: z) depend(in:z)
      z = z + 1
      !$omp end target
      !$omp end single
      !$omp end parallel
      ! Use 'accumulator' so it is not optimized away by the compiler.
      print *, accumulator
      r = z
    end subroutine foo

!CHECK: ======= FORTRAN Test passed! =======
!CHECK: foo(5) returned 6 , expected 6
end program main
