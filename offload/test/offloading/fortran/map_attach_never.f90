! This checks that attach never prevents pointer attachment when specified.
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

    correct = .true.
    n = 10

    do index = 1, n
        a(index) = 10
        b(index) = 20
    end do

    map_ptr => a

    ! This should map a and map_ptr to device, and attach map_ptr
    ! to a (as it is assigned to it above).
    !$omp target enter data map(ref_ptr_ptee, to: map_ptr)

    map_ptr => b

    ! As "b" hasn't been mapped to device yet, the first time it's mapped will
    ! be when map_ptr is re-mapped (implicitly or explicitly), the default behavior
    ! when LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS is switched off would force attachment
    ! of map_ptr to b as we've assigned it above. To prevent this and test the never
    ! attachment, we can apply attach(never), which prevents this reattachment from
    ! occurring
    !$omp target map(to: index) map(tofrom: correct) map(attach(never): map_ptr)
        do index = 1, n
            if (map_ptr(index) /= 10) then
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
