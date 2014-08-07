#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# void append_definitions(string new_flag);
# - appends new_flag to cpp_flags list
macro(append_definitions new_flag)
    list(APPEND local_cpp_flags "${new_flag}")
endmacro()

function(append_cpp_flags input_cpp_flags)
    set(local_cpp_flags)

    append_definitions("-D USE_ITT_BUILD")
    append_definitions("-D KMP_ARCH_STR=\"\\\\\"${legal_arch}\\\\\"\"")
    append_definitions("-D BUILD_I8")
    append_definitions("-D KMP_LIBRARY_FILE=\\\\\"${lib_file}\\\\\"") # yes... you need 5 backslashes...
    append_definitions("-D KMP_VERSION_MAJOR=${version}")
    
    # customize to 128 bytes for ppc64
    if(${PPC64})
    	append_definitions("-D CACHE_LINE=128")
    else()
    	append_definitions("-D CACHE_LINE=64")
    endif()
    
    append_definitions("-D KMP_ADJUST_BLOCKTIME=1")
    append_definitions("-D BUILD_PARALLEL_ORDERED")
    append_definitions("-D KMP_ASM_INTRINS")
    if(${USE_ITT_NOTIFY})
        append_definitions("-D USE_ITT_NOTIFY=1")
    else()
        append_definitions("-D USE_ITT_NOTIFY=0")
        append_definitions("-D INTEL_NO_ITTNOTIFY_API")
    endif()
    append_definitions("-D INTEL_ITTNOTIFY_PREFIX=__kmp_itt_")

    #####################
    # Windows definitions
    if(${WINDOWS})
        append_definitions("-D _CRT_SECURE_NO_WARNINGS")
        append_definitions("-D _CRT_SECURE_NO_DEPRECATE")
        append_definitions("-D _WINDOWS")
        append_definitions("-D _WINNT")
        append_definitions("-D _WIN32_WINNT=0x0501")
        append_definitions("-D KMP_WIN_CDECL")
        append_definitions("-D _USRDLL")
        if(${DEBUG_BUILD})
            append_definitions("-D _ITERATOR_DEBUG_LEVEL=0")
        endif()
    else() # Other than windows... (Unix based systems, Intel(R) Many Integrated Core Architecture (Intel(R) MIC Architecture), and Mac)
        append_definitions("-D _GNU_SOURCE")
        append_definitions("-D _REENTRANT")
        append_definitions("-D BUILD_TV")
        append_definitions("-D USE_CBLKDATA")
        if(NOT "${version}" STREQUAL "4")
            append_definitions("-D KMP_GOMP_COMPAT")
        endif()
    endif()

    #######################################
    # Intel(R) MIC Architecture definitions
    if(${MIC})
        append_definitions("-D KMP_TDATA_GTID")
    else() # Other than Intel(R) MIC Architecture...
        append_definitions("-D USE_LOAD_BALANCE")
    endif()

    ##################
    # Unix definitions
    if(${LINUX})
        append_definitions("-D KMP_TDATA_GTID")
    endif()

    ##################################
    # Other conditional definitions
    append_definitions("-D KMP_USE_ASSERT")
    append_definitions("-D GUIDEDLL_EXPORTS") 
    if(${STUBS_LIBRARY}) 
        append_definitions("-D KMP_STUB") 
    endif()
    if(${DEBUG_BUILD} OR ${RELWITHDEBINFO_BUILD}) 
        append_definitions("-D KMP_DEBUG") 
    endif()
    if(${DEBUG_BUILD})
        append_definitions("-D _DEBUG")
        append_definitions("-D BUILD_DEBUG")
    endif()
    if(${STATS_GATHERING})
        append_definitions("-D KMP_STATS_ENABLED=1")
    else()
        append_definitions("-D KMP_STATS_ENABLED=0")
    endif()

    # OpenMP version flags
    set(have_omp_50 0)
    set(have_omp_41 0)
    set(have_omp_40 0)
    set(have_omp_30 0)
    if(${omp_version} EQUAL 50 OR ${omp_version} GREATER 50)
        set(have_omp_50 1)
    endif()
    if(${omp_version} EQUAL 41 OR ${omp_version} GREATER 41)
        set(have_omp_41 1)
    endif()
    if(${omp_version} EQUAL 40 OR ${omp_version} GREATER 40)
        set(have_omp_40 1)
    endif()
    if(${omp_version} EQUAL 30 OR ${omp_version} GREATER 30)
        set(have_omp_30 1)
    endif()
    append_definitions("-D OMP_50_ENABLED=${have_omp_50}")
    append_definitions("-D OMP_41_ENABLED=${have_omp_41}")
    append_definitions("-D OMP_40_ENABLED=${have_omp_40}")
    append_definitions("-D OMP_30_ENABLED=${have_omp_30}")

    # Architectural definitions
    if(${INTEL64} OR ${IA32})
        if(${USE_ADAPTIVE_LOCKS})
            append_definitions("-D KMP_USE_ADAPTIVE_LOCKS=1")
        else()
            append_definitions("-D KMP_USE_ADAPTIVE_LOCKS=0")
        endif()
        append_definitions("-D KMP_DEBUG_ADAPTIVE_LOCKS=0")
    else()
        append_definitions("-D KMP_USE_ADAPTIVE_LOCKS=0")
        append_definitions("-D KMP_DEBUG_ADAPTIVE_LOCKS=0")
    endif()
    set(${input_cpp_flags} "${${input_cpp_flags}}" "${local_cpp_flags}" "${USER_CPP_FLAGS}" "$ENV{CPPFLAGS}" PARENT_SCOPE)
endfunction()

