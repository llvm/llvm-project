# This file holds Intel(R) C Compiler / Intel(R) C++ Compiler / Intel(R) Fortran Compiler (icc/icpc/icl.exe/ifort) dependent flags
# The flag types are:
#   2) C/C++ Compiler flags
#   4) Linker flags

# icc has a 128-bit floating point type called _Quad.  Always compile with 128-bit floating point if it exists.
unset(COMPILER_SUPPORTS_QUAD_PRECISION CACHE)
set(COMPILER_SUPPORTS_QUAD_PRECISION true CACHE BOOL "Does the compiler support a 128-bit floating point type?")
set(COMPILER_QUAD_TYPE _Quad)

#########################################################
# icc C/C++ Compiler flags
function(append_compiler_specific_c_and_cxx_flags input_c_flags input_cxx_flags)
    set(local_c_flags)
    set(local_cxx_flags)
    if(${WINDOWS})
        
        append_c_flags("-TP") # Tells the compiler to process a file as a C++ source file.
        append_cxx_flags("-EHsc") # Enable C++ exception handling.
        append_c_and_cxx_flags("-nologo") # Turn off tool banner.
        append_c_and_cxx_flags("-W3") # Enables diagnostics for remarks, warnings, and errors. 
                                      # Additional warnings are also enabled above level 2 warnings.
        append_c_and_cxx_flags("-WX") # Change all Warnings to Errors
        append_c_and_cxx_flags("-GS") # Lets you control the threshold at which the stack checking routine is called or not called.
        append_c_and_cxx_flags("-Qoption,cpp,--extended_float_types") # Enabled _Quad type.
        if(${IA32})
           append_c_and_cxx_flags("-arch:ia32") # Tells the compiler which features it may target (ia32)
           append_c_and_cxx_flags("-Oy-") # equivalent to -fno-omit-frame-pointer
        endif()
        append_c_and_cxx_flags("-Qlong_double") # enable long double
        append_c_and_cxx_flags("-Qdiag-disable:177") # Disable warning: "... declared but never referenced"
        if(${IA32})
            append_c_and_cxx_flags("-Qsafeseh") # Registers exception handlers for safe exception handling.
        endif()
        if(${RELEASE_BUILD} OR ${RELWITHDEBINFO_BUILD})
            append_c_and_cxx_flags("-Qinline-min-size=1") # Specifies the upper limit for the size of what the inliner considers to be a small routine.
        else()
            append_c_and_cxx_flags("-Od") # Disables all optimizations.
            append_c_and_cxx_flags("-RTC1") # Enables run-time checks of the stack frame, and enables run-time checks for unintialized variables.
            append_c_and_cxx_flags("-MTd") # Tells the linker to search for unresolved references in a multithreaded, static run-time library.
        endif()
    else()
        append_c_and_cxx_flags("-Wsign-compare") # warn on sign comparisons
        append_c_and_cxx_flags("-Werror") # Changes all warnings to errors.
        append_c_and_cxx_flags("-Qoption,cpp,--extended_float_types") # Enabled _Quad type.
        append_c_and_cxx_flags("-fno-exceptions") # Exception handling table generation is disabled.
        append_c_and_cxx_flags("-x c++") # Compile C files as C++ files
        if(${MIC})
            append_c_and_cxx_flags("-mmic") # Build Intel(R) MIC Architecture native code
            append_c_and_cxx_flags("-ftls-model=initial-exec") # Changes the thread local storage (TLS) model. Generates a restrictive, optimized TLS code. 
                                                               # To use this setting, the thread-local variables accessed must be defined in one of the 
                                                               # modules available to the program.
            append_c_and_cxx_flags("-opt-streaming-stores never") # Disables generation of streaming stores for optimization.
            append_c_and_cxx_flags("-sox") # Tells the compiler to save the compilation options and version number 
                                           # in the executable file. It also lets you choose whether to include 
                                           # lists of certain functions.
        elseif(${LINUX})
            append_c_and_cxx_flags("-Werror") # Changes all warnings to errors.
            append_c_and_cxx_flags("-sox") 
            if(${IA32})
                append_c_and_cxx_flags("-falign-stack=maintain-16-byte") # Tells the compiler the stack alignment to use on entry to routines.
                append_c_and_cxx_flags("-mia32")  # Tells the compiler which features it may target (ia32)
            endif()
        endif()
    endif()
    # CMake prefers the /MD flags when compiling Windows sources, but libiomp5 needs to use /MT instead
    # So we replace these /MD instances with /MT within the CMAKE_*_FLAGS variables and put that out to the CACHE.
    # replace_md_with_mt() is in HelperFunctions.cmake
    if(${WINDOWS})
        replace_md_with_mt(CMAKE_C_FLAGS)
        replace_md_with_mt(CMAKE_C_FLAGS_RELEASE)
        replace_md_with_mt(CMAKE_C_FLAGS_RELWITHDEBINFO)
        replace_md_with_mt(CMAKE_C_FLAGS_DEBUG)
        replace_md_with_mt(CMAKE_CXX_FLAGS)
        replace_md_with_mt(CMAKE_CXX_FLAGS_RELEASE)
        replace_md_with_mt(CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        replace_md_with_mt(CMAKE_CXX_FLAGS_DEBUG)
    endif()
    set(${input_c_flags}   ${${input_c_flags}}   "${local_c_flags}" PARENT_SCOPE)
    set(${input_cxx_flags} ${${input_cxx_flags}} "${local_cxx_flags}" PARENT_SCOPE)
endfunction()

#########################################################
# icc Linker flags
function(append_compiler_specific_linker_flags input_ld_flags input_ld_flags_libs)
    set(local_ld_flags)
    set(local_ld_flags_libs)
    if(${WINDOWS})
        # Have icc use link.exe directly when Windows
        set(CMAKE_C_CREATE_SHARED_LIBRARY "link.exe /out:<TARGET> <LINK_FLAGS> <OBJECTS> <LINK_LIBRARIES>" PARENT_SCOPE)
        set(CMAKE_SHARED_LINKER_FLAGS "$ENV{LDFLAGS}" CACHE STRING "Linker Flags" FORCE)
        append_linker_flags("-nologo") # Turn off tool banner.
        append_linker_flags("-dll") 
        append_linker_flags("-WX:NO")
        append_linker_flags("-incremental:no")
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
        if(${DEBUG_BUILD} OR ${RELWITHDEBINFO_BUILD})
            if(NOT "${pdb_file}" STREQUAL "")
                append_linker_flags("-debug")
                append_linker_flags("-pdb:${pdb_file}")
            endif()
        else()
            if(NOT "${pdb_file}" STREQUAL "")
                append_linker_flags("-debug")
                append_linker_flags("-pdb:${pdb_file}")
                append_linker_flags("-pdbstripped:${pdb_file}.stripped")
            endif()
        endif()
        if(NOT "${imp_file}" STREQUAL "")
            append_linker_flags("-implib:${lib_file}${lib}")
        endif()
        if(NOT "${def_file}" STREQUAL "")
            append_linker_flags("-def:${def_file}")
        endif()
    elseif(${MAC})
        append_linker_flags("-no-intel-extensions")
        if(NOT ${STUBS_LIBRARY})
            append_linker_flags_library("-pthread") # link in pthread library
            append_linker_flags_library("-ldl") # link in libdl (dynamic loader library)
        endif()
        if(${STATS_GATHERING})
            append_linker_flags_library("-Wl,-lstdc++") # link in standard c++ library (stats-gathering needs it)
        endif()
    elseif(${MIC})
        append_linker_flags("-mmic") # enable MIC linking
        append_linker_flags("-static-intel") # Causes Intel-provided libraries to be linked in statically. 
        append_linker_flags("-no-intel-extensions") # Enables or disables all Intel C and Intel C++ language extensions.
    else()
        append_linker_flags("-static-intel") # Causes Intel-provided libraries to be linked in statically.
        append_linker_flags("-Werror") # Warnings become errors
        if(${IA32})
            append_linker_flags_library("-lirc_pic") # link in libirc_pic
        endif()
    endif()

    set(${input_ld_flags}      ${${input_ld_flags}}      "${local_ld_flags}"       PARENT_SCOPE)
    set(${input_ld_flags_libs} ${${input_ld_flags_libs}} "${local_ld_flags_libs}"  PARENT_SCOPE)
endfunction()

