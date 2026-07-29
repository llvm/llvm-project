program logical_kinds
    implicit none

    logical(1) :: bool_one
    logical(2) :: bool_two
    logical(4) :: bool_four
    logical(8) :: bool_eight


    bool_one   = .true.
    bool_two   = .false.
    bool_four  = .true.
    bool_eight = .false.

    print *, "Done" ! Breakpoint here

end program logical_kinds