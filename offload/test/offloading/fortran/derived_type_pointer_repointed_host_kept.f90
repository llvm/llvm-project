! Verify that an attached pointer component can be repointed on the host without
! being restored back to its old association by target-region end processing.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-run-and-check-generic

program main
  implicit none

  type :: dtype
    integer, pointer :: p(:) => null()
    integer :: x
  end type dtype

  integer, target :: a(1, 2)
  type(dtype), target :: d

  a(1, 1) = 1
  a(1, 2) = 2

  if (repointed_host_pointer_kept(.true.)) then
    print *, "implicit kept"
  else
    print *, "implicit REVERTED"
    stop 1
  end if

  if (repointed_host_pointer_kept(.false.)) then
    print *, "present kept"
  else
    print *, "present REVERTED"
    stop 1
  end if

contains

  logical function repointed_host_pointer_kept(use_implicit)
    logical, intent(in) :: use_implicit

    d%x = 0
    d%p => a(:, 1)

    !$omp target enter data map(to: a, d, d%p)

    d%p => a(:, 2)

    if (use_implicit) then
      !$omp target
        d%x = 42
      !$omp end target
    else
      !$omp target map(present, alloc: d)
        d%x = 42
      !$omp end target
    end if

    repointed_host_pointer_kept = associated(d%p, a(:, 2))

    d%p => a(:, 1)
    !$omp target exit data map(release: d%p)
    !$omp target exit data map(delete: d, a)
    nullify(d%p)
  end function repointed_host_pointer_kept

end program main

! CHECK: implicit kept
! CHECK: present kept
