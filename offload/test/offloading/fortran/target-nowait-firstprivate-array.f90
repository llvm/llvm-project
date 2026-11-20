! REQUIRES: flang, flang-rt, amdgpu
! RUN: %libomptarget-compile-fortran-generic -O0
! RUN: %libomptarget-run-generic | %fcheck-generic
! RUN: %libomptarget-compile-fortran-generic -O2
! RUN: %libomptarget-run-generic | %fcheck-generic

module firstprivate_arrays
  implicit none
  integer, parameter :: n = 8, launches = 8
  integer :: values(n, launches), tokens(launches)
contains
  subroutine launch(k)
    integer, intent(in) :: k
    integer :: local(n), i

    do i = 1, n
      local(i) = 100 * k + i
    end do
    !$omp target teams distribute parallel do nowait firstprivate(local) &
    !$omp& map(present, alloc: values)
    do i = 1, n
      values(i, k) = local(i)
    end do
    ! The target task must use its copy after this array is overwritten and
    ! the subroutine returns.
    local = -1
  end subroutine

  subroutine launch_depend(k)
    integer, intent(in) :: k
    integer :: local(n), i

    do i = 1, n
      local(i) = 100 * k + i
    end do
    !$omp target teams distribute parallel do nowait depend(inout: tokens(k)) &
    !$omp& firstprivate(local) map(present, alloc: values)
    do i = 1, n
      values(i, k) = local(i)
    end do
    local = -1
  end subroutine

  subroutine check_values()
    integer :: i, k
    do k = 1, launches
      do i = 1, n
        if (values(i, k) /= 100 * k + i) stop 1
      end do
    end do
  end subroutine
end module

program test
  use firstprivate_arrays
  use omp_lib
  implicit none
  integer :: k, done(launches)
  logical :: initial_device

  !$omp target map(from: initial_device)
    initial_device = omp_is_initial_device()
  !$omp end target
  if (initial_device) stop 2

  values = 0
  !$omp target enter data map(to: values)
  !$omp parallel num_threads(2)
  !$omp single
    do k = 1, launches
      call launch(k)
    end do
    !$omp taskwait
  !$omp end single
  !$omp end parallel
  !$omp target update from(values)
  call check_values()
  print *, "PASS: firstprivate arrays with nowait"
  ! CHECK: PASS: firstprivate arrays with nowait

  values = 0
  done = 0
  !$omp target update to(values)
  !$omp parallel num_threads(2)
  !$omp single
    do k = 1, launches
      call launch_depend(k)
      !$omp task depend(in: tokens(k)) firstprivate(k) shared(done)
        done(k) = 1
      !$omp end task
    end do
    !$omp taskwait
  !$omp end single
  !$omp end parallel
  !$omp target update from(values)
  call check_values()
  if (any(done /= 1)) stop 3
  print *, "PASS: firstprivate arrays with nowait and depend"
  ! CHECK: PASS: firstprivate arrays with nowait and depend
  !$omp target exit data map(delete: values)
end program
