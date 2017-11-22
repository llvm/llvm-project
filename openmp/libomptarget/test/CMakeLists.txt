# CMakeLists.txt file for unit testing OpenMP Library
include(FindPythonInterp)
include(CheckTypeSize)
if(NOT PYTHONINTERP_FOUND)
  libomptarget_warning_say("Could not find Python.")
  libomptarget_warning_say("The check-libomptarget target will not be available!")
  return()
endif()

set(LIBOMPTARGET_TEST_CFLAGS "" CACHE STRING
  "Extra compiler flags to send to the test compiler")

if(LIBOMPTARGET_CMAKE_BUILD_TYPE MATCHES debug)
  set(LIBOMPTARGET_DEBUG True)
else()
  set(LIBOMPTARGET_DEBUG False)
endif()

if(${LIBOMPTARGET_STANDALONE_BUILD})
  # Make sure we can use the console pool for recent cmake and ninja > 1.5
  if(CMAKE_VERSION VERSION_LESS 3.1.20141117)
    set(cmake_3_2_USES_TERMINAL)
  else()
    set(cmake_3_2_USES_TERMINAL USES_TERMINAL)
  endif()
  set(LIBOMPTARGET_TEST_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING
    "C compiler to use for testing OpenMP offloading library")
  set(LIBOMPTARGET_TEST_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING
    "C++ compiler to use for testing OpenMP offloading library")

  if (NOT(${LIBOMPTARGET_TEST_C_COMPILER} MATCHES "clang" AND ${LIBOMPTARGET_TEST_CXX_COMPILER} MATCHES "clang"))
    libomptarget_say("Can only test with Clang compiler!")
    libomptarget_warning_say("The check-libomptarget target will not be available!")
    return()
  endif()

  execute_process(
    COMMAND ${LIBOMPTARGET_TEST_C_COMPILER} --version
    OUTPUT_VARIABLE TEST_COMPILER_VERSION)
  string(REGEX MATCH "version ([0-9.]+)" TEST_COMPILER_VERSION ${TEST_COMPILER_VERSION})
  if (NOT(TEST_COMPILER_VERSION))
    libomptarget_say("Unable to determine Clang compiler version!")
    libomptarget_warning_say("The check-libomptarget target will not be available!")
    return()
  endif()
  set(TEST_COMPILER_VERSION ${CMAKE_MATCH_1})
  if (TEST_COMPILER_VERSION VERSION_LESS 6.0.0)
    libomptarget_say("Clang compiler version does not implement all codegen, please update to 6.0.0!")
    libomptarget_warning_say("The check-libomptarget target will not be available!")
    return()
  endif()

  set(LIBOMPTARGET_TEST_OPENMP_FLAG -fopenmp CACHE STRING
    "OpenMP compiler flag to use for testing OpenMP offloading library")
  find_program(LIBOMPTARGET_LLVM_LIT_EXECUTABLE
    NAMES llvm-lit lit.py lit
    PATHS ${OPENMP_LLVM_TOOLS_DIR})
  if(NOT LIBOMPTARGET_LLVM_LIT_EXECUTABLE)
    libomptarget_say("Cannot find llvm-lit.")
    libomptarget_say("Please put llvm-lit in your PATH or set LIBOMPTARGET_LLVM_LIT_EXECUTABLE to its full path or point OPENMP_LLVM_TOOLS_DIR to its directory")
    libomptarget_warning_say("The check-libomptarget target will not be available!")
    return()
  endif()
  
  find_program(LIBOMPTARGET_FILECHECK_EXECUTABLE
    NAMES FileCheck
    PATHS ${OPENMP_LLVM_TOOLS_DIR})
  if(NOT LIBOMPTARGET_FILECHECK_EXECUTABLE)
    libomptarget_say("Cannot find FileCheck.")
    libomptarget_say("Please put FileCheck in your PATH or set LIBOMPTARGET_FILECHECK_EXECUTABLE to its full path or point OPENMP_LLVM_TOOLS_DIR to its directory")
    libomptarget_warning_say("The check-libomptarget target will not be available!")
    return()
  endif()
  
  # Set lit arguments
  # The -j 1 lets the actual tests run with the entire machine.
  # We have one test thread that spawns the tests serially.  This allows
  # Each test to use the entire machine.
  set(LIBOMPTARGET_LIT_ARGS_DEFAULT "-sv --show-unsupported --show-xfail -j 1")
  if(MSVC OR XCODE)
    set(LIBOMPTARGET_LIT_ARGS_DEFAULT "${LIBOMPTARGET_LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LIBOMPTARGET_LIT_ARGS "${LIBOMPTARGET_LIT_ARGS_DEFAULT}" CACHE STRING
    "Default options for lit")
  separate_arguments(LIBOMPTARGET_LIT_ARGS)
  add_custom_target(check-libomptarget
    COMMAND ${PYTHON_EXECUTABLE} ${LIBOMPTARGET_LLVM_LIT_EXECUTABLE} ${LIBOMPTARGET_LIT_ARGS} ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS omptarget
    COMMENT "Running libomptarget tests"
    ${cmake_3_2_USES_TERMINAL}
  )
  
  set(LIBOMPTARGET_OPENMP_HEADER_FOLDER "${CMAKE_CURRENT_BINARY_DIR}/../../runtime/src" CACHE STRING
    "Path to folder containing omp.h")
  set(LIBOMPTARGET_OPENMP_HOST_RTL_FOLDER "${CMAKE_CURRENT_BINARY_DIR}/../../runtime/src" CACHE STRING
    "Path to folder containing libomp.so")
else()
  # LLVM source tree build, test just-built clang
  if(NOT MSVC)
    set(LIBOMPTARGET_TEST_C_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
    set(LIBOMPTARGET_TEST_CXX_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++)
    set(LIBOMPTARGET_FILECHECK_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/FileCheck)
  else()
    libomptarget_warning_say("Not prepared to run tests on Windows systems.")
  endif()
  set(LIBOMPTARGET_TEST_OPENMP_FLAG -fopenmp=libomp)
  # Use add_lit_testsuite() from LLVM CMake. This also depends on OpenMP
  # implementation because it uses omp.h.
  add_lit_testsuite(check-libomptarget
    "Running libomptarget tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS omptarget omp
  )
  
  set(LIBOMPTARGET_OPENMP_HEADER_FOLDER "${LIBOMPTARGET_BINARY_DIR}/../runtime/src")
endif()

# Configure the lit.site.cfg.in file
set(AUTO_GEN_COMMENT "## Autogenerated by libomptarget configuration.\n# Do not edit!")
configure_file(lit.site.cfg.in lit.site.cfg @ONLY)

