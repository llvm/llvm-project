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

# void append_ev_flags(string new_flag);
# - appends new_flag to ev_flags list
macro(append_ev_flags new_flag)
    list(APPEND local_ev_flags "${new_flag}")
endmacro()

# void append_gd_flags(string new_flag);
# - appends new_flag to gd_flags list
macro(append_gd_flags new_flag)
    list(APPEND local_gd_flags "${new_flag}")
endmacro()

include(HelperFunctions) # for set_legal_type(), set_legal_arch()

# Perl expand-vars.pl flags
function(set_ev_flags input_ev_flags)
    set(local_ev_flags)
    set_legal_type("${lib_type}" legal_type)
    set_legal_arch("${arch}" legal_arch)
    # need -D Revision="\$Revision" to show up
    append_ev_flags("-D Revision=\"\\\\$$Revision\"") 
    append_ev_flags("-D Date=\"\\\\$$Date\"")
    append_ev_flags("-D KMP_TYPE=\"${legal_type}\"")
    append_ev_flags("-D KMP_ARCH=\"${legal_arch}\"")
    append_ev_flags("-D KMP_VERSION_MAJOR=${version}")
    append_ev_flags("-D KMP_VERSION_MINOR=0")
    append_ev_flags("-D KMP_VERSION_BUILD=${build_number}")
    append_ev_flags("-D KMP_BUILD_DATE=\"${date}\"")
    append_ev_flags("-D KMP_TARGET_COMPILER=12")
    if(${DEBUG_BUILD} OR ${RELWITHDEBINFO_BUILD})
        append_ev_flags("-D KMP_DIAG=1")
        append_ev_flags("-D KMP_DEBUG_INFO=1")
    else()
        append_ev_flags("-D KMP_DIAG=0")
        append_ev_flags("-D KMP_DEBUG_INFO=0")
    endif()
    if(${omp_version} EQUAL 40)
        append_ev_flags("-D OMP_VERSION=201307")
    elseif(${omp_version} EQUAL 30)
        append_ev_flags("-D OMP_VERSION=201107")
    else()
        append_ev_flags("-D OMP_VERSION=200505")
    endif()
    set(${input_ev_flags} "${local_ev_flags}" PARENT_SCOPE)
endfunction()

function(set_gd_flags input_gd_flags)
    set(local_gd_flags)
    if(${IA32})
        append_gd_flags("-D arch_32")
    elseif(${INTEL64})
        append_gd_flags("-D arch_32e")
    else()
        append_gd_flags("-D arch_${arch}")
    endif()
    if(${NORMAL_LIBRARY})
        append_gd_flags("-D norm")
    elseif(${PROFILE_LIBRARY})
        append_gd_flags("-D prof")
    elseif(${STUBS_LIBRARY})
        append_gd_flags("-D stub")
    endif()
    if(${omp_version} GREATER 40 OR ${omp_version} EQUAL 40)
        append_gd_flags("-D OMP_40")
    endif()
    if(${omp_version} GREATER 30 OR ${omp_version} EQUAL 30)
        append_gd_flags("-D OMP_30")
    endif()
    if(NOT "${version}" STREQUAL "4")
        append_gd_flags("-D msvc_compat")
    endif()
    if(${DEBUG_BUILD} OR ${RELWITHDEBINFO_BUILD})
        append_gd_flags("-D KMP_DEBUG")
    endif()
    if(${COMPILER_SUPPORTS_QUAD_PRECISION})
        append_gd_flags("-D HAVE_QUAD")
    endif()
    set(${input_gd_flags} "${local_gd_flags}" PARENT_SCOPE)
endfunction()
