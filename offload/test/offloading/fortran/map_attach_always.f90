
! This checks that attach always forces attachment.
! NOTE: We have to make sure the old default auto attach behaviour is off to
! yield the correct results for this test. Otherwise the second target will
! be treated as if we'd had the attach always specified!
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=0 %libomptarget-run-generic 2>&1 | %fcheck-generic
program main
    implicit none
    integer,  pointer :: map_ptr(:)
    integer, target :: a(10)
    integer, target :: b(10)
    integer :: index, n
    logical :: correct

    n = 10
    correct = .true.

    do index = 1, n
        a(index) = 10
        b(index) = 20
    end do

    map_ptr => a

   ! This should map a,b and map_ptr to device, and attach map_ptr
   ! to a (as it is assigned to it above), and as b is already on
   ! device running through target.
   !$omp target enter data map(ref_ptr_ptee, to: map_ptr)
   !$omp target enter data map(to: b, a)

    !$omp target map(to: index) map(tofrom: correct)
        do index = 1, n
            if (map_ptr(index) /= 10) then
                correct = .false.
            endif
        end do
    !$omp end target

    map_ptr => b

    ! No attach always to force re-attachment, so we should still
    ! be attached to "a"
    !$omp target map(to: index) map(tofrom: correct)
        do index = 1, n
            if (map_ptr(index) /= 10) then
                correct = .false.
            endif
        end do
    !$omp end target

    !$omp target map(to: index) map(attach(always): map_ptr) map(tofrom: correct)
        do index = 1, n
            if (map_ptr(index) /= 20) then
                correct = .false.
            endif
        end do
    !$omp end target

    if (correct .NEQV. .true.) then
        print*, "Failed!"
        stop 1
    endif

    print*, "Passed!"
end program

!CHECK: Passed!
