cmake_minimum_required(VERSION 3.7)

# We're building the compiler's CRT library, so cmake cannot yet build + link a simple test application.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(nextcrt)

if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "Defaulting to debug build. To specify build type: cmake -DCMAKE_BUILD_TYPE=[Release|Debug|RelWithDebInfo]")
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(-O3 -Wall -Wextra -Wfloat-conversion)

set(NEXTCRT_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib/${NEXTCRT_LIBDIR_SUBDIR} CACHE STRING "")
set(NEXTCRT_INSTALL_LIBRARY_DIR lib/${NEXTCRT_LIBDIR_SUBDIR} CACHE STRING "")

add_subdirectory(next32_builtins)
add_subdirectory(next32rt)
add_subdirectory(libm-stub)
