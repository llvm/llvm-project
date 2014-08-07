# This file holds GNU (gcc/g++) specific compiler dependent flags
# The flag types are:
#   1) Fortran Compiler flags

#########################################################
# Fortran Compiler flags (for creating .mod files)
function(append_fortran_compiler_specific_fort_flags input_fort_flags)
    set(local_fort_flags)
    if(${IA32})
        append_fort_flags("-m32")
        append_fort_flags("-msse2")
    endif()
    set(${input_fort_flags} ${${input_fort_flags}} "${local_fort_flags}" PARENT_SCOPE)
endfunction()
