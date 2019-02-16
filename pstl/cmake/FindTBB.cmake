#===-- FindTBB.cmake -----------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

include(FindPackageHandleStandardArgs)

if (NOT TBB_FIND_COMPONENTS)
    set(TBB_FIND_COMPONENTS tbb tbbmalloc)
    foreach (_tbb_component ${TBB_FIND_COMPONENTS})
        set(TBB_FIND_REQUIRED_${_tbb_component} 1)
    endforeach()
endif()

find_path(_tbb_include_dir tbb/tbb.h)
if (_tbb_include_dir)
    file(READ "${_tbb_include_dir}/tbb/tbb_stddef.h" _tbb_stddef LIMIT 2048)
    string(REGEX REPLACE ".*#define TBB_VERSION_MAJOR ([0-9]+).*" "\\1" _tbb_ver_major "${_tbb_stddef}")
    string(REGEX REPLACE ".*#define TBB_VERSION_MINOR ([0-9]+).*" "\\1" _tbb_ver_minor "${_tbb_stddef}")
    string(REGEX REPLACE ".*#define TBB_INTERFACE_VERSION ([0-9]+).*" "\\1" TBB_INTERFACE_VERSION "${_tbb_stddef}")

    set(TBB_VERSION "${_tbb_ver_major}.${_tbb_ver_minor}")

    unset(_tbb_stddef)
    unset(_tbb_ver_major)
    unset(_tbb_ver_minor)

    foreach (_tbb_component ${TBB_FIND_COMPONENTS})
        find_library(_tbb_release_lib ${_tbb_component})
        if (_tbb_release_lib)
            set(TBB_${_tbb_component}_FOUND 1)

            add_library(TBB::${_tbb_component} SHARED IMPORTED)
            list(APPEND TBB_IMPORTED_TARGETS TBB::${_tbb_component})

            set(_tbb_lib_suffix)
            if (UNIX AND NOT APPLE)
                set(_tbb_lib_suffix ".2")
            endif()

            set_target_properties(TBB::${_tbb_component} PROPERTIES
                                  IMPORTED_CONFIGURATIONS       "RELEASE"
                                  IMPORTED_LOCATION_RELEASE     "${_tbb_release_lib}${_tbb_lib_suffix}"
                                  INTERFACE_INCLUDE_DIRECTORIES "${_tbb_include_dir}")

            find_library(_tbb_debug_lib ${_tbb_component}_debug)
            if (_tbb_debug_lib)
                set_target_properties(TBB::${_tbb_component} PROPERTIES
                                      IMPORTED_CONFIGURATIONS "RELEASE;DEBUG"
                                      IMPORTED_LOCATION_DEBUG "${_tbb_debug_lib}${_tbb_lib_suffix}")
            endif()
            unset(_tbb_debug_lib CACHE)
            unset(_tbb_lib_suffix)
        endif()
        unset(_tbb_release_lib CACHE)
    endforeach()
endif()
unset(_tbb_include_dir CACHE)

find_package_handle_standard_args(TBB
                                  REQUIRED_VARS TBB_IMPORTED_TARGETS
                                  HANDLE_COMPONENTS
                                  VERSION_VAR TBB_VERSION)
