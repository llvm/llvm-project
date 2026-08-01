program integer_kinds
    implicit none

    integer(1) :: tiny_int
    integer(2) :: short_int
    integer(4) :: normal_int
    integer(8) :: huge_int

    tiny_int   = 127
    short_int  = 32767
    normal_int = 2147483647
    huge_int   = 9223372036854775807_8 

    print *, "Done" ! Breakpoint here

end program integer_kinds