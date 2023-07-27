subroutine increment_at(c_index, arr) bind(C, name="increment_at")
    use ISO_C_BINDING
    integer (C_INT), dimension(*), intent(inout) :: arr
    integer (C_INT), value :: c_index
    arr(c_index+1) = arr(c_index+1) + 1
end subroutine
