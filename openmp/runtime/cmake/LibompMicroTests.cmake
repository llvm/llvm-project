#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

# The following micro-tests are small tests to perform on the library just created.
# There are currently five micro-tests:
# (1) test-touch
#  - Compile and run a small program using newly created libomp library
#  - Fails if test-touch.c does not compile or if test-touch.c does not run after compilation
#  - Program dependencies: gcc or g++, grep, bourne shell
#  - Available for all Unix,Mac,Windows builds.  Not available on Intel(R) MIC Architecture builds.
# (2) test-relo
#  - Tests dynamic libraries for position-dependent code (can not have any position dependent code)
#  - Fails if TEXTREL is in output of readelf -d libomp.so command
#  - Program dependencies: readelf, grep, bourne shell
#  - Available for Unix, Intel(R) MIC Architecture dynamic library builds. Not available otherwise.
# (3) test-execstack
#  - Tests if stack is executable
#  - Fails if stack is executable. Should only be readable and writable. Not exectuable.
#  - Program dependencies: perl, readelf
#  - Available for Unix dynamic library builds. Not available otherwise.
# (4) test-instr (Intel(R) MIC Architecutre only)
#  - Tests Intel(R) MIC Architecture libraries for valid instruction set
#  - Fails if finds invalid instruction for Intel(R) MIC Architecture (wasn't compiled with correct flags)
#  - Program dependencies: perl, objdump
#  - Available for Intel(R) MIC Architecture and i386 builds. Not available otherwise.
# (5) test-deps
#  - Tests newly created libomp for library dependencies
#  - Fails if sees a dependence not listed in td_exp variable below
#  - Program dependencies: perl, (unix)readelf, (mac)otool[64], (windows)link.exe
#  - Available for Unix,Mac,Windows, Intel(R) MIC Architecture dynamic builds and Windows
#    static builds. Not available otherwise.

# get library location
if(WIN32)
  get_target_property(LIBOMP_OUTPUT_DIRECTORY omp RUNTIME_OUTPUT_DIRECTORY)
  get_target_property(LIBOMPIMP_OUTPUT_DIRECTORY ompimp ARCHIVE_OUTPUT_DIRECTORY)
  if(NOT LIBOMPIMP_OUTPUT_DIRECTORY)
    set(LIBOMPIMP_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
else()
  get_target_property(LIBOMP_OUTPUT_DIRECTORY omp LIBRARY_OUTPUT_DIRECTORY)
endif()
if(NOT LIBOMP_OUTPUT_DIRECTORY)
  set(LIBOMP_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

# test-touch
find_program(LIBOMP_SHELL sh)
if(WIN32)
  if(LIBOMP_SHELL)
    set(libomp_test_touch_targets test-touch-md/.success test-touch-mt/.success)
  endif()
  # pick test-touch compiler
  set(libomp_test_touch_compiler ${CMAKE_C_COMPILER})
  # test-touch compilation flags
  libomp_append(libomp_test_touch_cflags /nologo)
  libomp_append(libomp_test_touch_libs ${LIBOMPIMP_OUTPUT_DIRECTORY}/${LIBOMP_IMP_LIB_FILE})
  if(${IA32})
    libomp_append(libomp_test_touch_ldflags /safeseh)
  endif()
else() # (Unix based systems, Intel(R) MIC Architecture, and Mac)
  if(LIBOMP_SHELL)
    set(libomp_test_touch_targets test-touch-rt/.success)
  endif()
  # pick test-touch compiler
  if(${LIBOMP_USE_STDCPPLIB})
    set(libomp_test_touch_compiler ${CMAKE_CXX_COMPILER})
  else()
    set(libomp_test_touch_compiler ${CMAKE_C_COMPILER})
  endif()
  # test-touch compilation flags
  libomp_append(libomp_test_touch_libs "${CMAKE_THREAD_LIBS_INIT}")
  if(${IA32})
    libomp_append(libomp_test_touch_cflags -m32 LIBOMP_HAVE_M32_FLAG)
  endif()
  libomp_append(libomp_test_touch_libs ${LIBOMP_OUTPUT_DIRECTORY}/${LIBOMP_LIB_FILE})
  libomp_append(libomp_test_touch_libs "${LIBOMP_HWLOC_LIBRARY}" LIBOMP_USE_HWLOC)
  if(APPLE)
    set(libomp_test_touch_env "DYLD_LIBRARY_PATH=.:${LIBOMP_OUTPUT_DIRECTORY}:$ENV{DYLD_LIBRARY_PATH}")
    libomp_append(libomp_test_touch_ldflags "-Wl,-rpath,${LIBOMP_HWLOC_LIBRARY_DIR}" LIBOMP_USE_HWLOC)
  else()
    set(libomp_test_touch_env "LD_LIBRARY_PATH=.:${LIBOMP_OUTPUT_DIRECTORY}:$ENV{LD_LIBRARY_PATH}")
    libomp_append(libomp_test_touch_ldflags "-Wl,-rpath=${LIBOMP_HWLOC_LIBRARY_DIR}" LIBOMP_USE_HWLOC)
  endif()
endif()
macro(libomp_test_touch_recipe test_touch_dir)
  set(libomp_test_touch_dependencies ${LIBOMP_SRC_DIR}/test-touch.c omp)
  set(libomp_test_touch_exe ${test_touch_dir}/test-touch${CMAKE_EXECUTABLE_SUFFIX})
  set(libomp_test_touch_obj ${test_touch_dir}/test-touch${CMAKE_C_OUTPUT_EXTENSION})
  if(WIN32)
    if(${RELEASE_BUILD} OR ${RELWITHDEBINFO_BUILD})
      if(${test_touch_dir} MATCHES "test-touch-mt")
        libomp_append(libomp_test_touch_cflags /MT)
      else()
        libomp_append(libomp_test_touch_cflags /MD)
      endif()
    else()
      if(${test_touch_dir} MATCHES "test-touch-mt")
        libomp_append(libomp_test_touch_cflags /MTd)
      else()
        libomp_append(libomp_test_touch_cflags /MDd)
      endif()
    endif()
    set(libomp_test_touch_out_flags -Fe${libomp_test_touch_exe} -Fo${libomp_test_touch_obj})
    list(APPEND libomp_test_touch_dependencies ompimp)
  else()
    set(libomp_test_touch_out_flags -o ${libomp_test_touch_exe})
  endif()
  add_custom_command(
    OUTPUT  ${test_touch_dir}/.success ${libomp_test_touch_exe} ${libomp_test_touch_obj}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${test_touch_dir}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${test_touch_dir}/*
    COMMAND ${libomp_test_touch_compiler} ${libomp_test_touch_out_flags} ${libomp_test_touch_cflags}
      ${LIBOMP_SRC_DIR}/test-touch.c ${libomp_test_touch_ldflags} ${libomp_test_touch_libs}
    COMMAND ${LIBOMP_SHELL} -c \"${libomp_test_touch_env} ${libomp_test_touch_exe}\"
    COMMAND ${CMAKE_COMMAND} -E touch ${test_touch_dir}/.success
    DEPENDS ${libomp_test_touch_dependencies}
  )
endmacro()
libomp_append(libomp_test_touch_env "KMP_VERSION=1")
add_custom_target(libomp-test-touch DEPENDS ${libomp_test_touch_targets})
if(WIN32)
  libomp_test_touch_recipe(test-touch-mt)
  libomp_test_touch_recipe(test-touch-md)
else()
  libomp_test_touch_recipe(test-touch-rt)
endif()

# test-relo
add_custom_target(libomp-test-relo DEPENDS test-relo/.success)
add_custom_command(
  OUTPUT  test-relo/.success test-relo/readelf.log
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/test-relo
  COMMAND readelf -d ${LIBOMP_OUTPUT_DIRECTORY}/${LIBOMP_LIB_FILE} > test-relo/readelf.log
  COMMAND grep -e TEXTREL test-relo/readelf.log \; test $$? -eq 1
  COMMAND ${CMAKE_COMMAND} -E touch test-relo/.success
  DEPENDS omp
)

# test-execstack
add_custom_target(libomp-test-execstack DEPENDS test-execstack/.success)
add_custom_command(
  OUTPUT  test-execstack/.success
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/test-execstack
  COMMAND ${PERL_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/check-execstack.pl
    --arch=${LIBOMP_PERL_SCRIPT_ARCH} ${LIBOMP_OUTPUT_DIRECTORY}/${LIBOMP_LIB_FILE}
  COMMAND ${CMAKE_COMMAND} -E touch test-execstack/.success
  DEPENDS omp
)

# test-instr
add_custom_target(libomp-test-instr DEPENDS test-instr/.success)
add_custom_command(
  OUTPUT  test-instr/.success
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/test-instr
  COMMAND ${PERL_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/check-instruction-set.pl --os=${LIBOMP_PERL_SCRIPT_OS}
    --arch=${LIBOMP_PERL_SCRIPT_ARCH} --show --mic-arch=${LIBOMP_MIC_ARCH} ${LIBOMP_OUTPUT_DIRECTORY}/${LIBOMP_LIB_FILE}
  COMMAND ${CMAKE_COMMAND} -E touch test-instr/.success
  DEPENDS omp ${LIBOMP_TOOLS_DIR}/check-instruction-set.pl
)

# test-deps
add_custom_target(libomp-test-deps DEPENDS test-deps/.success)
set(libomp_expected_library_deps)
if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  set(libomp_expected_library_deps libc.so.7 libthr.so.3 libm.so.5)
  libomp_append(libomp_expected_library_deps libhwloc.so.5 LIBOMP_USE_HWLOC)
elseif(CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  set(libomp_expected_library_deps libc.so.12 libpthread.so.1 libm.so.0)
  libomp_append(libomp_expected_library_deps libhwloc.so.5 LIBOMP_USE_HWLOC)
elseif(CMAKE_SYSTEM_NAME MATCHES "DragonFly")
  set(libomp_expected_library_deps libc.so.8 libpthread.so.0 libm.so.4)
  libomp_append(libomp_expected_library_deps libhwloc.so.5 LIBOMP_USE_HWLOC)
elseif(APPLE)
  set(libomp_expected_library_deps /usr/lib/libSystem.B.dylib)
elseif(WIN32)
  set(libomp_expected_library_deps kernel32.dll)
  libomp_append(libomp_expected_library_deps psapi.dll LIBOMP_OMPT_SUPPORT)
else()
  if(${MIC})
    set(libomp_expected_library_deps libc.so.6 libpthread.so.0 libdl.so.2)
    if("${LIBOMP_MIC_ARCH}" STREQUAL "knf")
      libomp_append(libomp_expected_library_deps ld-linux-l1om.so.2)
      libomp_append(libomp_expected_library_deps libgcc_s.so.1)
    elseif("${LIBOMP_MIC_ARCH}" STREQUAL "knc")
      libomp_append(libomp_expected_library_deps ld-linux-k1om.so.2)
    endif()
  else()
    set(libomp_expected_library_deps libdl.so.2 libgcc_s.so.1)
    if(${IA32})
      libomp_append(libomp_expected_library_deps libc.so.6)
      libomp_append(libomp_expected_library_deps ld-linux.so.2)
    elseif(${INTEL64})
      libomp_append(libomp_expected_library_deps libc.so.6)
      libomp_append(libomp_expected_library_deps ld-linux-x86-64.so.2)
    elseif(${ARM})
      libomp_append(libomp_expected_library_deps libc.so.6)
      libomp_append(libomp_expected_library_deps libffi.so.6)
      libomp_append(libomp_expected_library_deps libffi.so.5)
      libomp_append(libomp_expected_library_deps ld-linux-armhf.so.3)
    elseif(${PPC64})
      libomp_append(libomp_expected_library_deps libc.so.6)
      libomp_append(libomp_expected_library_deps ld64.so.1)
    elseif(${MIPS} OR ${MIPS64})
      libomp_append(libomp_expected_library_deps libc.so.6)
      libomp_append(libomp_expected_library_deps ld.so.1)
    endif()
    libomp_append(libomp_expected_library_deps libpthread.so.0 IF_FALSE STUBS_LIBRARY)
    libomp_append(libomp_expected_library_deps libhwloc.so.5 LIBOMP_USE_HWLOC)
  endif()
  libomp_append(libomp_expected_library_deps libstdc++.so.6 LIBOMP_USE_STDCPPLIB)
  libomp_append(libomp_expected_library_deps libm.so.6 LIBOMP_STATS)
endif()
# Perl script expects comma separated list
string(REPLACE ";" "," libomp_expected_library_deps "${libomp_expected_library_deps}")
add_custom_command(
  OUTPUT  test-deps/.success
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/test-deps
  COMMAND ${PERL_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/check-depends.pl --os=${LIBOMP_PERL_SCRIPT_OS}
    --arch=${LIBOMP_PERL_SCRIPT_ARCH} --expected="${libomp_expected_library_deps}" ${LIBOMP_OUTPUT_DIRECTORY}/${LIBOMP_LIB_FILE}
  COMMAND ${CMAKE_COMMAND} -E touch test-deps/.success
  DEPENDS omp ${LIBOMP_TOOLS_DIR}/check-depends.pl
)
