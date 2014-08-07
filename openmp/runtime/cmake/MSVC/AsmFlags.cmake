# This file holds Microsoft Visual Studio dependent flags
# The flag types are:
#   1) Assembly flags

#########################################################
# Assembly flags
function(append_assembler_specific_asm_flags input_asm_flags)
    set(local_asm_flags)
    append_asm_flags("-nologo") # Turn off tool banner.
    if(${IA32})
        append_asm_flags("-safeseh") # Registers exception handlers for safe exception handling.
        append_asm_flags("-coff") # Generates common object file format (COFF) type of object module. 
                                  # Generally required for Win32 assembly language development.
        append_asm_flags("-D _M_IA32")
    elseif(${INTEL64})
        append_asm_flags("-D _M_AMD64")
    endif()
    # CMake prefers the /MD flags when compiling Windows sources, but libiomp5 needs to use /MT instead
    # So we replace these /MD instances with /MT within the CMAKE_*_FLAGS variables and put that out to the CACHE.
    # replace_md_with_mt() is in HelperFunctions.cmake
    replace_md_with_mt(CMAKE_ASM_MASM_FLAGS)
    replace_md_with_mt(CMAKE_ASM_MASM_FLAGS_RELEASE)
    replace_md_with_mt(CMAKE_ASM_MASM_FLAGS_RELWITHDEBINFO)
    replace_md_with_mt(CMAKE_ASM_MASM_FLAGS_DEBUG)
    set(${input_asm_flags} ${${input_asm_flags}} "${local_asm_flags}" PARENT_SCOPE)
endfunction()
