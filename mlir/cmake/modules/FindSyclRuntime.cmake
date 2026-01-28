# CMake find_package() module for SYCL Runtime
#
# Example usage:
#
# find_package(SyclRuntime)
#
# If successful, the following variables will be defined:
# SyclRuntime_FOUND
# SyclRuntime_INCLUDE_DIRS
# SyclRuntime_LIBRARY
# SyclRuntime_LIBRARIES_DIR
#

include(FindPackageHandleStandardArgs)

if(NOT DEFINED ENV{CMPLR_ROOT})
    message(WARNING "Please make sure to install Intel DPC++ Compiler and run setvars.(sh/bat)")
    message(WARNING "You can download standalone Intel DPC++ Compiler from https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#compilers")
else()
    get_filename_component(ONEAPI_VER "$ENV{CMPLR_ROOT}" NAME)
    if(ONEAPI_VER VERSION_LESS 2024.0)
        if(LINUX OR ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux"))
            set(SyclRuntime_ROOT "$ENV{CMPLR_ROOT}/linux")
        elseif(WIN32)
            set(SyclRuntime_ROOT "$ENV{CMPLR_ROOT}/windows")
        endif()
    else()
        set(SyclRuntime_ROOT "$ENV{CMPLR_ROOT}")
    endif()
    list(APPEND SyclRuntime_INCLUDE_DIRS "${SyclRuntime_ROOT}/include")
    list(APPEND SyclRuntime_INCLUDE_DIRS "${SyclRuntime_ROOT}/include/sycl")

    set(SyclRuntime_LIBRARY_DIR "${SyclRuntime_ROOT}/lib")

    message(STATUS "SyclRuntime_LIBRARY_DIR: ${SyclRuntime_LIBRARY_DIR}")
    find_library(SyclRuntime_LIBRARY
        NAMES sycl
        PATHS ${SyclRuntime_LIBRARY_DIR}
        NO_DEFAULT_PATH
        )
endif()

if(SyclRuntime_LIBRARY)
    set(SyclRuntime_FOUND TRUE)
    if(NOT TARGET SyclRuntime::SyclRuntime)
        add_library(SyclRuntime::SyclRuntime INTERFACE IMPORTED)
        set_target_properties(SyclRuntime::SyclRuntime
            PROPERTIES INTERFACE_LINK_LIBRARIES "${SyclRuntime_LIBRARY}"
      )
      set_target_properties(SyclRuntime::SyclRuntime
          PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SyclRuntime_INCLUDE_DIRS}"
      )
    endif()
else()
    set(SyclRuntime_FOUND FALSE)
endif()

find_package_handle_standard_args(SyclRuntime
    REQUIRED_VARS
        SyclRuntime_FOUND
        SyclRuntime_INCLUDE_DIRS
        SyclRuntime_LIBRARY
        SyclRuntime_LIBRARY_DIR
    HANDLE_COMPONENTS
)

mark_as_advanced(SyclRuntime_LIBRARY SyclRuntime_INCLUDE_DIRS)

if(SyclRuntime_FOUND)
    find_package_message(SyclRuntime "Found SyclRuntime: ${SyclRuntime_LIBRARY}" "")
else()
    find_package_message(SyclRuntime "Could not find SyclRuntime" "")
endif()
