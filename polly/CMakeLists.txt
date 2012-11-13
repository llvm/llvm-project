# Check if this is a in tree build.
if (NOT DEFINED LLVM_MAIN_SRC_DIR)
  project(Polly)
  cmake_minimum_required(VERSION 2.8)

  # Where is LLVM installed?
  set(LLVM_INSTALL_ROOT "" CACHE PATH "Root of LLVM install.")
  # Check if the LLVM_INSTALL_ROOT valid.
  if( NOT EXISTS ${LLVM_INSTALL_ROOT}/include/llvm )
    message(FATAL_ERROR "LLVM_INSTALL_ROOT (${LLVM_INSTALL_ROOT}) is not a valid LLVM installation.")
  endif(NOT EXISTS ${LLVM_INSTALL_ROOT}/include/llvm)
  #FileCheck is not install by default, warn the user to Copy FileCheck
  if( NOT EXISTS ${LLVM_INSTALL_ROOT}/bin/FileCheck
      OR NOT EXISTS ${LLVM_INSTALL_ROOT}/bin/not)
    message(WARNING "FileCheck or not are requred by running regress tests, "
                    "but they are not installed! Please copy it to "
                    "${LLVM_INSTALL_ROOT}/bin.")
  endif(NOT EXISTS ${LLVM_INSTALL_ROOT}/bin/FileCheck
        OR NOT EXISTS ${LLVM_INSTALL_ROOT}/bin/not)
  # Add the llvm header path.
  include_directories(${LLVM_INSTALL_ROOT}/include/)

  # Get the system librarys that will link into LLVM.
  function(get_system_libs return_var)
    # Returns in `return_var' a list of system libraries used by LLVM.
    if( NOT MSVC )
      if( MINGW )
        set(system_libs ${system_libs} imagehlp psapi)
      elseif( CMAKE_HOST_UNIX )
        if( HAVE_LIBDL )
          set(system_libs ${system_libs} ${CMAKE_DL_LIBS})
        endif()
        if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
          set(system_libs ${system_libs} pthread)
        endif()
      endif( MINGW )
    endif( NOT MSVC )
    set(${return_var} ${system_libs} PARENT_SCOPE)
  endfunction(get_system_libs)

  # Now set the header paths.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --includedir
                  OUTPUT_VARIABLE LLVM_INCLUDE_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  include_directories( ${LLVM_INCLUDE_DIR} )

  # Get the TARGET_TRIPLE
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --host-target
                  OUTPUT_VARIABLE TARGET_TRIPLE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # And then set the cxx flags.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --cxxflags
                  OUTPUT_VARIABLE LLVM_CXX_FLAGS
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${LLVM_CXX_FLAGS})
endif(NOT DEFINED LLVM_MAIN_SRC_DIR)

set(POLLY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POLLY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  )

include("polly_macros")

# Add appropriate flags for GCC
if (CMAKE_COMPILER_IS_GNUCXX)
  # FIXME: Turn off exceptions, RTTI:
  # -fno-exceptions -fno-rtti
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings -fno-exceptions -fno-rtti")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

# Add path for custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${POLLY_SOURCE_DIR}/cmake")

FIND_PACKAGE(Cloog REQUIRED)
FIND_PACKAGE(Isl REQUIRED)
FIND_PACKAGE(Gmp REQUIRED)
FIND_PACKAGE(Pluto)

option(POLLY_ENABLE_GPGPU_CODEGEN "Enable GPGPU code generation feature" OFF)
if (POLLY_ENABLE_GPGPU_CODEGEN)
  # Do not require CUDA, as GPU code generation test cases can be run without
  # a cuda library.
  FIND_PACKAGE(CUDA)
  set(GPU_CODEGEN TRUE)
endif(POLLY_ENABLE_GPGPU_CODEGEN)

option(POLLY_ENABLE_OPENSCOP "Enable Openscop library for scop import/export" ON)
if (POLLY_ENABLE_OPENSCOP)
  FIND_PACKAGE(OpenScop)
endif(POLLY_ENABLE_OPENSCOP)

option(POLLY_ENABLE_SCOPLIB "Enable SCoPLib library for scop import/export" ON)
if (POLLY_ENABLE_SCOPLIB)
  FIND_PACKAGE(SCoPLib)
endif(POLLY_ENABLE_SCOPLIB)

if (CLOOG_FOUND)
  INCLUDE_DIRECTORIES( ${CLOOG_INCLUDE_DIR} )
endif(CLOOG_FOUND)
if (PLUTO_FOUND)
  INCLUDE_DIRECTORIES( ${PLUTO_INCLUDE_DIR} )
endif(PLUTO_FOUND)
INCLUDE_DIRECTORIES( ${ISL_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${GMP_INCLUDE_DIR} )

# Support GPGPU code generation if the library is available.
if (CUDALIB_FOUND)
  INCLUDE_DIRECTORIES( ${CUDALIB_INCLUDE_DIR} )
endif(CUDALIB_FOUND)

# Support OpenScop export/import if the library is available.
if (OPENSCOP_FOUND)
  INCLUDE_DIRECTORIES( ${OPENSCOP_INCLUDE_DIR} )
endif(OPENSCOP_FOUND)
if (SCOPLIB_FOUND)
  INCLUDE_DIRECTORIES( ${SCOPLIB_INCLUDE_DIR} )
endif(SCOPLIB_FOUND)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/JSON/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

install(DIRECTORY include
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  )

add_definitions( -D_GNU_SOURCE )

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
# TODO: docs.


configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/include/polly/Config/config.h.cmake
                ${POLLY_BINARY_DIR}/include/polly/Config/config.h )

