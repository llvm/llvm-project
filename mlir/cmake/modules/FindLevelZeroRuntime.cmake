# CMake find_package() module for level-zero
#
# Example usage:
#
# find_package(LevelZero)
#
# If successful, the following variables will be defined:
# LevelZero_FOUND
# LevelZero_INCLUDE_DIRS
# LevelZero_LIBRARY
# LevelZero_LIBRARIES_DIR
#
# By default, the module searches the standard paths to locate the "ze_api.h"
# and the ze_loader shared library. When using a custom level-zero installation,
# the environment variable "LEVEL_ZERO_DIR" should be specified telling the
# module to get the level-zero library and headers from that location.

include(FindPackageHandleStandardArgs)

# Search path priority
# 1. CMake Variable LEVEL_ZERO_DIR
# 2. Environment Variable LEVEL_ZERO_DIR
if(NOT LEVEL_ZERO_DIR)
    if(DEFINED ENV{LEVEL_ZERO_DIR})
        set(LEVEL_ZERO_DIR "$ENV{LEVEL_ZERO_DIR}")
    endif()
endif()

if(LEVEL_ZERO_DIR)
    find_path(LevelZeroRuntime_INCLUDE_DIR
        NAMES level_zero/ze_api.h
        PATHS ${LEVEL_ZERO_DIR}/include
        NO_DEFAULT_PATH
    )

    if(LINUX)
        find_library(LevelZeroRuntime_LIBRARY
            NAMES ze_loader
            PATHS ${LEVEL_ZERO_DIR}/lib
            ${LEVEL_ZERO_DIR}/lib/x86_64-linux-gnu
            NO_DEFAULT_PATH
        )
    else()
        find_library(LevelZeroRuntime_LIBRARY
            NAMES ze_loader
            PATHS ${LEVEL_ZERO_DIR}/lib
            NO_DEFAULT_PATH
        )
    endif()
else()
    find_path(LevelZeroRuntime_INCLUDE_DIR
        NAMES level_zero/ze_api.h
    )

    find_library(LevelZeroRuntime_LIBRARY
        NAMES ze_loader
    )
endif()

# Compares the two version string that are supposed to be in x.y.z format
# and reports if the argument VERSION_STR1 is greater than or equal than
# version_str2. The strings are compared lexicographically after conversion to
# lists of equal lengths, with the shorter string getting zero-padded.
function(compare_versions VERSION_STR1 VERSION_STR2 OUTPUT)
    # Convert the strings to list
    string(REPLACE "." ";" VL1 ${VERSION_STR1})
    string(REPLACE "." ";" VL2 ${VERSION_STR2})

    # get lengths of both lists
    list(LENGTH VL1 VL1_LEN)
    list(LENGTH VL2 VL2_LEN)
    set(LEN ${VL1_LEN})

    # If they differ in size pad the shorter list with 0s
    if(VL1_LEN GREATER VL2_LEN)
        math(EXPR DIFF "${VL1_LEN} - ${VL2_LEN}" OUTPUT_FORMAT DECIMAL)
        foreach(IDX RANGE 1 ${DIFF} 1)
            list(APPEND VL2 "0")
        endforeach()
    elseif(VL2_LEN GREATER VL2_LEN)
        math(EXPR DIFF "${VL1_LEN} - ${VL2_LEN}" OUTPUT_FORMAT DECIMAL)
        foreach(IDX RANGE 1 ${DIFF} 1)
            list(APPEND VL2 "0")
        endforeach()
        set(LEN ${VL2_LEN})
    endif()
    math(EXPR LEN_SUB_ONE "${LEN}-1")
    foreach(IDX RANGE 0 ${LEN_SUB_ONE} 1)
        list(GET VL1 ${IDX} VAL1)
        list(GET VL2 ${IDX} VAL2)

        if(${VAL1} GREATER ${VAL2})
            set(${OUTPUT} TRUE PARENT_SCOPE)
            break()
        elseif(${VAL1} LESS ${VAL2})
            set(${OUTPUT} FALSE PARENT_SCOPE)
            break()
        else()
            set(${OUTPUT} TRUE PARENT_SCOPE)
        endif()
    endforeach()
endfunction(compare_versions)

# Creates a small function to run and extract the LevelZero loader version.
function(get_l0_loader_version)
    set(L0_VERSIONEER_SRC
        [====[
        #include <iostream>
        #include <level_zero/loader/ze_loader.h>
        #include <string>
        int main() {
            ze_result_t result;
            std::string loader("loader");
            zel_component_version_t *versions;
            size_t size = 0;
            result = zeInit(0);
            if (result != ZE_RESULT_SUCCESS) {
                std::cerr << "Failed to init ze driver" << std::endl;
                return -1;
            }
            zelLoaderGetVersions(&size, nullptr);
            versions = new zel_component_version_t[size];
            zelLoaderGetVersions(&size, versions);
            for (size_t i = 0; i < size; i++) {
                if (loader.compare(versions[i].component_name) == 0) {
                    std::cout << versions[i].component_lib_version.major << "."
                              << versions[i].component_lib_version.minor << "."
                              << versions[i].component_lib_version.patch;
                    break;
                }
            }
            delete[] versions;
            return 0;
        }
        ]====]
    )

    set(L0_VERSIONEER_FILE ${CMAKE_BINARY_DIR}/temp/l0_versioneer.cpp)

    file(WRITE ${L0_VERSIONEER_FILE} "${L0_VERSIONEER_SRC}")

    # We need both the directories in the include path as ze_loader.h
    # includes "ze_api.h" and not "level_zero/ze_api.h".
    list(APPEND INCLUDE_DIRS ${LevelZeroRuntime_INCLUDE_DIR})
    list(APPEND INCLUDE_DIRS ${LevelZeroRuntime_INCLUDE_DIR}/level_zero)
    list(JOIN INCLUDE_DIRS ";" INCLUDE_DIRS_STR)
    try_run(L0_VERSIONEER_RUN L0_VERSIONEER_COMPILE
        "${CMAKE_BINARY_DIR}"
        "${L0_VERSIONEER_FILE}"
        LINK_LIBRARIES ${LevelZeroRuntime_LIBRARY}
        CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES=${INCLUDE_DIRS_STR}"
        RUN_OUTPUT_VARIABLE L0_VERSION
    )

    if(${L0_VERSIONEER_COMPILE} AND(DEFINED L0_VERSIONEER_RUN))
        set(LevelZeroRuntime_VERSION ${L0_VERSION} PARENT_SCOPE)
        message(STATUS "Found Level Zero of version: ${L0_VERSION}")
    else()
        message(FATAL_ERROR
            "Could not compile a level-zero program to extract loader version"
        )
    endif()
endfunction(get_l0_loader_version)

if(LevelZeroRuntime_INCLUDE_DIR AND LevelZeroRuntime_LIBRARY)
    list(APPEND LevelZeroRuntime_LIBRARIES "${LevelZeroRuntime_LIBRARY}")
    list(APPEND LevelZeroRuntime_INCLUDE_DIRS ${LevelZeroRuntime_INCLUDE_DIR})

    if(OpenCL_FOUND)
        list(APPEND LevelZeroRuntime_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
    endif()

    cmake_path(GET LevelZeroRuntime_LIBRARY PARENT_PATH LevelZeroRuntime_LIBRARIES_PATH)
    set(LevelZeroRuntime_LIBRARIES_DIR ${LevelZeroRuntime_LIBRARIES_PATH})

    if(NOT TARGET LevelZeroRuntime::LevelZeroRuntime)
        add_library(LevelZeroRuntime::LevelZeroRuntime INTERFACE IMPORTED)
        set_target_properties(LevelZeroRuntime::LevelZeroRuntime
            PROPERTIES INTERFACE_LINK_LIBRARIES "${LevelZeroRuntime_LIBRARIES}"
        )
        set_target_properties(LevelZeroRuntime::LevelZeroRuntime
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LevelZeroRuntime_INCLUDE_DIRS}"
        )
    endif()
endif()

# Check if a specific version of Level Zero is required
if(LevelZeroRuntime_FIND_VERSION)
    get_l0_loader_version()
    set(VERSION_GT_FIND_VERSION FALSE)
    compare_versions(
        ${LevelZeroRuntime_VERSION}
        ${LevelZeroRuntime_FIND_VERSION}
        VERSION_GT_FIND_VERSION
    )

    if(${VERSION_GT_FIND_VERSION})
        set(LevelZeroRuntime_FOUND TRUE)
    else()
        set(LevelZeroRuntime_FOUND FALSE)
    endif()
else()
    set(LevelZeroRuntime_FOUND TRUE)
endif()

find_package_handle_standard_args(LevelZeroRuntime
    REQUIRED_VARS
    LevelZeroRuntime_FOUND
    LevelZeroRuntime_INCLUDE_DIRS
    LevelZeroRuntime_LIBRARY
    LevelZeroRuntime_LIBRARIES_DIR
    HANDLE_COMPONENTS
)
mark_as_advanced(LevelZeroRuntime_LIBRARY LevelZeroRuntime_INCLUDE_DIRS)

if(LevelZeroRuntime_FOUND)
    find_package_message(LevelZeroRuntime "Found LevelZero: ${LevelZeroRuntime_LIBRARY}"
        "(found version ${LevelZeroRuntime_VERSION})"
    )
else()
    find_package_message(LevelZeroRuntime "Could not find LevelZero" "")
endif()
