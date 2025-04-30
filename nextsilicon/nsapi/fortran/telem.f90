module nsapi
    use, intrinsic :: iso_c_binding, only: C_CHAR
    implicit none

    interface
        subroutine nsapi_telem_region_enter() bind(c, name='nsapi_telem_region_enter')
            use, intrinsic :: iso_c_binding, only: C_CHAR
        end subroutine nsapi_telem_region_enter

        subroutine nsapi_telem_region_exit() bind(c, name='nsapi_telem_region_exit')
            use, intrinsic :: iso_c_binding, only: C_CHAR
        end subroutine nsapi_telem_region_exit
    end interface
end module nsapi
