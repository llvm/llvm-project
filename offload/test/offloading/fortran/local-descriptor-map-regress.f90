! Small regression test that checks that we do not cause
! a runtime map error in cases where we are required to
! allocate a local variable for the fortran descriptor
! to store into and then load from it, done so by
! re-using the temporary local variable across all
! maps related to the mapped variable and associated
! local variable to make sure that each map does as
! it's intended to do with the original data. This
! prevents blobs of local descriptor data remaining
! attatched on device long after it's supposed to,
! which can cause weird map issues later in susbequent
! function invocations. However, it doesn't avoid a user
! shooting themselves in the foot by mapping data via enter
! and then not providing a corresponding exit.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
subroutine launchOne(n1, n2, ret)
    implicit none
    real, intent(out) :: ret
    real(4), dimension(n1,n2) :: sbuf31
    integer :: n1,n2
!$omp target enter data map(alloc:sbuf31)

!$omp target
      sbuf31(2, 2) = 10
!$omp end target

!$omp target update from(sbuf31)

   ret = sbuf31(2, 2)

!$omp target exit data map(delete:sbuf31)
end subroutine launchOne

subroutine launchTwo(N, ret)
    implicit none
    real, intent(inout) :: ret
    integer :: N
    real(4), dimension(N) :: p

!$omp target enter data map(to:p)

!$omp target
    p(8) = 20
!$omp end target

!$omp target update from(p)

ret = ret + p(8)

! intentional non-deletion, can trigger an illegal map
! issue in cases where the local map we store and load
! from for the variable is different across all maps.
! Not too sure why this is the thing that triggers the
! problem in general. It seems like it would be an issue
! made apparent with and without this statement commented,
! especially as the issue occurs with the enter and not the
! corresponding exit (from the runtime trace at least).
!!$omp target exit data map(delete:p)
end subroutine launchTwo

program reproducer
    implicit none
    integer :: N = 10
    integer :: nr = 10, nt = 10
    real :: output = 0

    call launchOne(nr, nt,  output)
    call launchTwo(N, output)

    print *, output
end program reproducer

! CHECK: 30
