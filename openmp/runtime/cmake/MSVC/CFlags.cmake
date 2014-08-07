# This file holds Microsoft Visual Studio dependent flags
# The flag types are:
#   1) C/C++ Compiler flags
#   2) Fortran Compiler flags

#########################################################
# Visual Studio C/C++ Compiler flags
function(append_compiler_specific_c_and_cxx_flags input_c_flags input_cxx_flags)
    set(local_c_flags)
    set(local_cxx_flags)
    append_c_flags("-TP") # Tells the compiler to process a file as a C++ source file.
    append_cxx_flags("-EHsc") # Enable C++ exception handling.
    append_c_and_cxx_flags("-W3") # Enables diagnostics for remarks, warnings, and errors. 
                                  # Additional warnings are also enabled above level 2 warnings.
    append_c_and_cxx_flags("-GS") # Lets you control the threshold at which the stack checking routine is called or not called.
    if(${IA32})
        append_c_and_cxx_flags("-arch:ia32") # Tells the compiler which features it may target (ia32)
        append_c_and_cxx_flags("-Oy-") # equivalent to -fno-omit-frame-pointer
    endif()
    # CMake prefers the /MD flags when compiling Windows sources, but libiomp5 needs to use /MT instead
    # So we replace these /MD instances with /MT within the CMAKE_*_FLAGS variables and put that out to the CACHE.
    # replace_md_with_mt() is in HelperFunctions.cmake
    replace_md_with_mt(CMAKE_C_FLAGS)
    replace_md_with_mt(CMAKE_C_FLAGS_RELEASE)
    replace_md_with_mt(CMAKE_C_FLAGS_RELWITHDEBINFO)
    replace_md_with_mt(CMAKE_C_FLAGS_DEBUG)
    replace_md_with_mt(CMAKE_CXX_FLAGS)
    replace_md_with_mt(CMAKE_CXX_FLAGS_RELEASE)
    replace_md_with_mt(CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    replace_md_with_mt(CMAKE_CXX_FLAGS_DEBUG)
    replace_md_with_mt(CMAKE_ASM_MASM_FLAGS)
    replace_md_with_mt(CMAKE_ASM_MASM_FLAGS_RELEASE)
    replace_md_with_mt(CMAKE_ASM_MASM_FLAGS_RELWITHDEBINFO)
    replace_md_with_mt(CMAKE_ASM_MASM_FLAGS_DEBUG)
    set(${input_c_flags}   ${${input_c_flags}}   "${local_c_flags}" PARENT_SCOPE)
    set(${input_cxx_flags} ${${input_cxx_flags}} "${local_cxx_flags}" PARENT_SCOPE)
endfunction()

#########################################################
# Visual Studio Linker flags
function(append_compiler_specific_linker_flags input_ld_flags input_ld_flags_libs)
    set(local_ld_flags)
    set(local_ld_flags_libs)
    append_linker_flags("-WX:NO")
    append_linker_flags("-version:${version}.0")
    append_linker_flags("-NXCompat")
    append_linker_flags("-DynamicBase") # This option modifies the header of an executable to indicate 
                                           # whether the application should be randomly rebased at load time.
    if(${IA32})
        append_linker_flags("-machine:i386")
        append_linker_flags("-safeseh")
    elseif(${INTEL64})
        append_linker_flags("-machine:amd64")
    endif()
    if(NOT "${def_file}" STREQUAL "")
        append_linker_flags("-def:${def_file}")
    endif()
    # Have Visual Studio use link.exe directly
    #set(CMAKE_C_CREATE_SHARED_LIBRARY "link.exe /out:<TARGET> <LINK_FLAGS> <OBJECTS> <LINK_LIBRARIES>" PARENT_SCOPE)
    #set(CMAKE_SHARED_LINKER_FLAGS "$ENV{LDLFAGS}" CACHE STRING "Linker Flags" FORCE)
    set(${input_ld_flags}      ${${input_ld_flags}}      "${local_ld_flags}"       PARENT_SCOPE)
    set(${input_ld_flags_libs} ${${input_ld_flags_libs}} "${local_ld_flags_libs}"  PARENT_SCOPE)
endfunction()

