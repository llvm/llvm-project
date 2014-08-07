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

macro(append_c_source_file new_c_file)
    list(APPEND local_c_source_files "${new_c_file}")
endmacro()

macro(append_cpp_source_file new_cpp_file)
    list(APPEND local_cpp_source_files "${new_cpp_file}")
endmacro()

macro(append_asm_source_file new_asm_file)
    list(APPEND local_asm_source_files "${new_asm_file}")
endmacro()

macro(append_imp_c_source_file new_import_c_file)
    list(APPEND local_imp_c_files "${new_import_c_file}")
endmacro()

# files are relative to the src directory

function(set_c_files input_c_source_files) 
    set(local_c_source_files "")
    append_c_source_file("kmp_ftn_cdecl.c")
    append_c_source_file("kmp_ftn_extra.c")
    append_c_source_file("kmp_version.c")
    if(${STUBS_LIBRARY})
        append_c_source_file("kmp_stub.c")
    else()
        append_c_source_file("kmp_alloc.c")
        append_c_source_file("kmp_atomic.c")
        append_c_source_file("kmp_csupport.c")
        append_c_source_file("kmp_debug.c")
        append_c_source_file("kmp_itt.c")
        append_c_source_file("kmp_environment.c")
        append_c_source_file("kmp_error.c")
        append_c_source_file("kmp_global.c")
        append_c_source_file("kmp_i18n.c")
        append_c_source_file("kmp_io.c")
        append_c_source_file("kmp_runtime.c")
        append_c_source_file("kmp_settings.c")
        append_c_source_file("kmp_str.c")
        append_c_source_file("kmp_tasking.c")
        append_c_source_file("kmp_taskq.c")
        append_c_source_file("kmp_threadprivate.c")
        append_c_source_file("kmp_utility.c")
        if(${USE_ITT_NOTIFY})
            append_c_source_file("thirdparty/ittnotify/ittnotify_static.c")
        endif()
        if(${WINDOWS})
            append_c_source_file("z_Windows_NT_util.c")
            append_c_source_file("z_Windows_NT-586_util.c")
        else()
            append_c_source_file("z_Linux_util.c")
            append_c_source_file("kmp_gsupport.c")
        endif()
    endif()
    set(${input_c_source_files} "${local_c_source_files}" PARENT_SCOPE)
endfunction()

function(set_cpp_files input_cpp_source_files) 
    set(local_cpp_source_files "")
    if(NOT ${STUBS_LIBRARY})
        #append_cpp_source_file("kmp_barrier.cpp")
        append_cpp_source_file("kmp_affinity.cpp")
        append_cpp_source_file("kmp_dispatch.cpp")
        append_cpp_source_file("kmp_lock.cpp")
        append_cpp_source_file("kmp_sched.cpp")
        if("${omp_version}" STREQUAL "40")
            append_cpp_source_file("kmp_taskdeps.cpp")
            append_cpp_source_file("kmp_cancel.cpp")
        endif()
        #if(${STATS_GATHERING})
        #   append_cpp_source_file("kmp_stats.cpp")
        #    append_cpp_source_file("kmp_stats_timing.cpp")
        #endif()
    endif()

    set(${input_cpp_source_files} "${local_cpp_source_files}" PARENT_SCOPE)
endfunction()


function(set_asm_files input_asm_source_files) 
    set(local_asm_source_files "")
    if(NOT ${STUBS_LIBRARY})
        if(${WINDOWS})
            append_asm_source_file("z_Windows_NT-586_asm.asm")
        else()
            append_asm_source_file("z_Linux_asm.s")
        endif()
    endif()
    set(${input_asm_source_files} "${local_asm_source_files}" PARENT_SCOPE)
endfunction()


function(set_imp_c_files input_imp_c_files)
    set(local_imp_c_files "")
    if(${WINDOWS})
        append_imp_c_source_file("kmp_import.c")
    endif()
    set(${input_imp_c_files} "${local_imp_c_files}" PARENT_SCOPE)
endfunction()
