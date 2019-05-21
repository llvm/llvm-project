#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

# Configure omp.h, kmp_config.h and omp-tools.h if necessary
configure_file(${LIBOMP_INC_DIR}/omp.h.var omp.h @ONLY)
configure_file(kmp_config.h.cmake kmp_config.h @ONLY)
if(${LIBOMP_OMPT_SUPPORT})
  configure_file(${LIBOMP_INC_DIR}/omp-tools.h.var omp-tools.h @ONLY)
endif()

# Generate message catalog files: kmp_i18n_id.inc and kmp_i18n_default.inc
add_custom_command(
  OUTPUT  kmp_i18n_id.inc
  COMMAND ${PERL_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/message-converter.pl --os=${LIBOMP_PERL_SCRIPT_OS}
    --prefix=kmp_i18n --enum=kmp_i18n_id.inc ${LIBOMP_SRC_DIR}/i18n/en_US.txt
  DEPENDS ${LIBOMP_SRC_DIR}/i18n/en_US.txt ${LIBOMP_TOOLS_DIR}/message-converter.pl
)
add_custom_command(
  OUTPUT  kmp_i18n_default.inc
  COMMAND ${PERL_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/message-converter.pl --os=${LIBOMP_PERL_SCRIPT_OS}
    --prefix=kmp_i18n --default=kmp_i18n_default.inc ${LIBOMP_SRC_DIR}/i18n/en_US.txt
  DEPENDS ${LIBOMP_SRC_DIR}/i18n/en_US.txt ${LIBOMP_TOOLS_DIR}/message-converter.pl
)

# Set the -D definitions for all sources
# UNICODE and _UNICODE are set in LLVM's CMake system.  They affect the
# ittnotify code and should only be set when compiling ittnotify_static.c
# on Windows (done below).
# TODO: Fix the UNICODE usage in ittnotify code for Windows.
remove_definitions(-DUNICODE -D_UNICODE)
libomp_get_definitions_flags(LIBOMP_CONFIGURED_DEFINITIONS_FLAGS)
add_definitions(${LIBOMP_CONFIGURED_DEFINITIONS_FLAGS})

# Set the -I includes for all sources
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${LIBOMP_SRC_DIR}
  ${LIBOMP_SRC_DIR}/i18n
  ${LIBOMP_INC_DIR}
  ${LIBOMP_SRC_DIR}/thirdparty/ittnotify
)
if(${LIBOMP_USE_HWLOC})
  include_directories(${LIBOMP_HWLOC_INSTALL_DIR}/include)
endif()

# Getting correct source files to build library
set(LIBOMP_CFILES)
set(LIBOMP_CXXFILES)
set(LIBOMP_ASMFILES)
if(${STUBS_LIBRARY})
  set(LIBOMP_CFILES kmp_stub.cpp)
else()
  # Get C++ files
  set(LIBOMP_CXXFILES
    kmp_alloc.cpp
    kmp_atomic.cpp
    kmp_csupport.cpp
    kmp_debug.cpp
    kmp_itt.cpp
    kmp_environment.cpp
    kmp_error.cpp
    kmp_global.cpp
    kmp_i18n.cpp
    kmp_io.cpp
    kmp_runtime.cpp
    kmp_settings.cpp
    kmp_str.cpp
    kmp_tasking.cpp
    kmp_threadprivate.cpp
    kmp_utility.cpp
    kmp_barrier.cpp
    kmp_wait_release.cpp
    kmp_affinity.cpp
    kmp_dispatch.cpp
    kmp_lock.cpp
    kmp_sched.cpp
  )
  if(WIN32)
    # Windows specific files
    libomp_append(LIBOMP_CXXFILES z_Windows_NT_util.cpp)
    libomp_append(LIBOMP_CXXFILES z_Windows_NT-586_util.cpp)
    libomp_append(LIBOMP_ASMFILES z_Windows_NT-586_asm.asm) # Windows assembly file
  else()
    # Unix specific files
    libomp_append(LIBOMP_CXXFILES z_Linux_util.cpp)
    libomp_append(LIBOMP_CXXFILES kmp_gsupport.cpp)
    libomp_append(LIBOMP_ASMFILES z_Linux_asm.S) # Unix assembly file
  endif()
  libomp_append(LIBOMP_CFILES thirdparty/ittnotify/ittnotify_static.c LIBOMP_USE_ITT_NOTIFY)
  libomp_append(LIBOMP_CXXFILES kmp_debugger.cpp LIBOMP_USE_DEBUGGER)
  libomp_append(LIBOMP_CXXFILES kmp_stats.cpp LIBOMP_STATS)
  libomp_append(LIBOMP_CXXFILES kmp_stats_timing.cpp LIBOMP_STATS)
  if(${LIBOMP_OMP_VERSION} GREATER 40 OR ${LIBOMP_OMP_VERSION} EQUAL 40)
    libomp_append(LIBOMP_CXXFILES kmp_taskdeps.cpp)
    libomp_append(LIBOMP_CXXFILES kmp_cancel.cpp)
  endif()
endif()
# Files common to stubs and normal library
libomp_append(LIBOMP_CXXFILES kmp_ftn_cdecl.cpp)
libomp_append(LIBOMP_CXXFILES kmp_ftn_extra.cpp)
libomp_append(LIBOMP_CXXFILES kmp_version.cpp)
libomp_append(LIBOMP_CXXFILES ompt-general.cpp IF_TRUE LIBOMP_OMPT_SUPPORT)
libomp_append(LIBOMP_CXXFILES tsan_annotations.cpp IF_TRUE LIBOMP_TSAN_SUPPORT)

set(LIBOMP_SOURCE_FILES ${LIBOMP_CFILES} ${LIBOMP_CXXFILES} ${LIBOMP_ASMFILES})
# For Windows, there is a resource file (.rc -> .res) that is also compiled
libomp_append(LIBOMP_SOURCE_FILES libomp.rc WIN32)

# Get compiler and assembler flags
libomp_get_cflags(LIBOMP_CONFIGURED_CFLAGS)
libomp_get_cxxflags(LIBOMP_CONFIGURED_CXXFLAGS)
libomp_get_asmflags(LIBOMP_CONFIGURED_ASMFLAGS)
# Set the compiler flags for each type of source
set_source_files_properties(${LIBOMP_CFILES} PROPERTIES COMPILE_FLAGS "${LIBOMP_CONFIGURED_CFLAGS}")
set_source_files_properties(${LIBOMP_CXXFILES} PROPERTIES COMPILE_FLAGS "${LIBOMP_CONFIGURED_CXXFLAGS}")
set_source_files_properties(${LIBOMP_ASMFILES} PROPERTIES COMPILE_FLAGS "${LIBOMP_CONFIGURED_ASMFLAGS}")
# Let the compiler handle the assembly files on Unix-like systems
if(NOT WIN32)
  set_source_files_properties(${LIBOMP_ASMFILES} PROPERTIES LANGUAGE C)
endif()

# Remove any cmake-automatic linking of the standard C++ library.
# We neither need (nor want) the standard C++ library dependency even though we compile c++ files.
if(NOT ${LIBOMP_USE_STDCPPLIB})
  set(LIBOMP_LINKER_LANGUAGE C)
  set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES)
else()
  set(LIBOMP_LINKER_LANGUAGE CXX)
endif()

# Add the OpenMP library
libomp_get_ldflags(LIBOMP_CONFIGURED_LDFLAGS)

add_library(omp ${LIBOMP_LIBRARY_KIND} ${LIBOMP_SOURCE_FILES})

set_target_properties(omp PROPERTIES
  PREFIX "" SUFFIX "" OUTPUT_NAME "${LIBOMP_LIB_FILE}"
  LINK_FLAGS "${LIBOMP_CONFIGURED_LDFLAGS}"
  LINKER_LANGUAGE ${LIBOMP_LINKER_LANGUAGE}
)

# Get the library's location within the build tree for the unit tester
if(NOT WIN32)
  get_target_property(LIBOMP_LIBRARY_DIR omp LIBRARY_OUTPUT_DIRECTORY)
else()
  get_target_property(LIBOMP_LIBRARY_DIR omp RUNTIME_OUTPUT_DIRECTORY)
endif()
if(NOT LIBOMP_LIBRARY_DIR)
  set(LIBOMP_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  set(LIBOMP_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
else()
  set(LIBOMP_LIBRARY_DIR ${LIBOMP_LIBRARY_DIR} PARENT_SCOPE)
endif()

# Add symbolic links to libomp
if(NOT WIN32)
  add_custom_command(TARGET omp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIBOMP_LIB_FILE}
      libgomp${LIBOMP_LIBRARY_SUFFIX}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIBOMP_LIB_FILE}
      libiomp5${LIBOMP_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${LIBOMP_LIBRARY_DIR}
  )
endif()

# Linking command will include libraries in LIBOMP_CONFIGURED_LIBFLAGS
libomp_get_libflags(LIBOMP_CONFIGURED_LIBFLAGS)
target_link_libraries(omp ${LIBOMP_CONFIGURED_LIBFLAGS} ${CMAKE_DL_LIBS})

# Create *.inc before compiling any sources
# objects depend on : .inc files
add_custom_target(libomp-needed-headers DEPENDS kmp_i18n_id.inc kmp_i18n_default.inc)
add_dependencies(omp libomp-needed-headers)

# Windows specific build rules
if(WIN32)
  configure_file(libomp.rc.var libomp.rc @ONLY)

  # Create .def and .rc file before compiling any sources
  add_custom_target(libomp-needed-windows-files DEPENDS ${LIBOMP_LIB_NAME}.def)
  add_dependencies(omp libomp-needed-windows-files)
  # z_Windows_NT-586_asm.asm requires definitions to be sent via command line
  # It only needs the architecutre macro and OMPT_SUPPORT=0|1
  libomp_append(LIBOMP_MASM_DEFINITIONS "-D_M_IA32" IF_TRUE IA32)
  libomp_append(LIBOMP_MASM_DEFINITIONS "-D_M_AMD64" IF_TRUE INTEL64)
  libomp_append(LIBOMP_MASM_DEFINITIONS "-DOMPT_SUPPORT" IF_TRUE_1_0 LIBOMP_OMPT_SUPPORT)
  libomp_list_to_string("${LIBOMP_MASM_DEFINITIONS}" LIBOMP_MASM_DEFINITIONS)
  set_property(SOURCE z_Windows_NT-586_asm.asm APPEND_STRING PROPERTY COMPILE_FLAGS " ${LIBOMP_MASM_DEFINITIONS}")
  set_source_files_properties(thirdparty/ittnotify/ittnotify_static.c PROPERTIES COMPILE_DEFINITIONS "UNICODE")

  # Create Windows import library
  # the import library is "re-linked" to include kmp_import.cpp which prevents
  # linking of both Visual Studio OpenMP and newly built OpenMP
  set_source_files_properties(kmp_import.cpp PROPERTIES COMPILE_FLAGS "${LIBOMP_CONFIGURED_CFLAGS}")
  set(LIBOMP_IMP_LIB_FILE ${LIBOMP_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(LIBOMP_GENERATED_IMP_LIB_FILENAME ${LIBOMP_LIB_FILE}${CMAKE_STATIC_LIBRARY_SUFFIX})
  set_target_properties(omp PROPERTIES
    VERSION ${LIBOMP_VERSION_MAJOR}.${LIBOMP_VERSION_MINOR} # uses /version flag
    IMPORT_PREFIX "" IMPORT_SUFFIX "" # control generated import library name when building omp
    ARCHIVE_OUTPUT_NAME ${LIBOMP_GENERATED_IMP_LIB_FILENAME}
  )
  # Get generated import library from creating omp
  get_target_property(LIBOMP_IMPORT_LIB_DIRECTORY omp ARCHIVE_OUTPUT_DIRECTORY)
  if(LIBOMP_IMPORT_LIB_DIRECTORY)
    set(LIBOMP_GENERATED_IMP_LIB ${LIBOMP_IMPORT_LIB_DIRECTORY}/${LIBOMP_GENERATED_IMP_LIB_FILENAME})
  else()
    set(LIBOMP_GENERATED_IMP_LIB ${CMAKE_CURRENT_BINARY_DIR}/${LIBOMP_GENERATED_IMP_LIB_FILENAME})
  endif()
  set_source_files_properties(${LIBOMP_GENERATED_IMP_LIB} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  # Create new import library that is just the previously created one + kmp_import.cpp
  add_library(ompimp STATIC ${LIBOMP_GENERATED_IMP_LIB} kmp_import.cpp)
  set_target_properties(ompimp PROPERTIES
    PREFIX "" SUFFIX "" OUTPUT_NAME "${LIBOMP_IMP_LIB_FILE}"
    LINKER_LANGUAGE C
  )
  add_dependencies(ompimp omp) # ensure generated import library is created first

  # Create def file to designate exported functions
  libomp_get_gdflags(LIBOMP_GDFLAGS) # generate-def.pl flags (Windows only)
  libomp_string_to_list("${LIBOMP_GDFLAGS}" LIBOMP_GDFLAGS)
  add_custom_command(
    OUTPUT  ${LIBOMP_LIB_NAME}.def
    COMMAND ${PERL_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/generate-def.pl ${LIBOMP_GDFLAGS}
      -o ${LIBOMP_LIB_NAME}.def ${CMAKE_CURRENT_SOURCE_DIR}/dllexports
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dllexports ${LIBOMP_TOOLS_DIR}/generate-def.pl
  )
endif()

# Building the Fortran module files
# One compilation step creates both omp_lib.mod and omp_lib_kinds.mod
if(${LIBOMP_FORTRAN_MODULES})
  configure_file(${LIBOMP_INC_DIR}/omp_lib.h.var omp_lib.h @ONLY)
  configure_file(${LIBOMP_INC_DIR}/omp_lib.f.var omp_lib.f @ONLY)
  configure_file(${LIBOMP_INC_DIR}/omp_lib.f90.var omp_lib.f90 @ONLY)
  # Workaround for gfortran to build modules with the
  # omp_sched_monotonic integer parameter
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(ADDITIONAL_Fortran_FLAGS "-fno-range-check")
  endif()
  add_custom_target(libomp-mod ALL DEPENDS omp_lib.mod omp_lib_kinds.mod)
  libomp_get_fflags(LIBOMP_CONFIGURED_FFLAGS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    set(LIBOMP_FORTRAN_SOURCE_FILE omp_lib.f90)
  else()
    set(LIBOMP_FORTRAN_SOURCE_FILE omp_lib.f)
  endif()
  add_custom_command(
    OUTPUT omp_lib.mod omp_lib_kinds.mod
    COMMAND ${CMAKE_Fortran_COMPILER} -c ${ADDITIONAL_Fortran_FLAGS}
            ${LIBOMP_CONFIGURED_FFLAGS} ${LIBOMP_FORTRAN_SOURCE_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LIBOMP_FORTRAN_SOURCE_FILE}
      ${CMAKE_CURRENT_BINARY_DIR}/omp_lib.h
  )
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES omp_lib${CMAKE_C_OUTPUT_EXTENSION})
endif()

# Move files to exports/ directory if requested
if(${LIBOMP_COPY_EXPORTS})
  include(LibompExports)
endif()

# Micro test rules for after library has been built (cmake/LibompMicroTests.cmake)
include(LibompMicroTests)
add_custom_target(libomp-micro-tests)
if(NOT ${MIC} AND NOT CMAKE_CROSSCOMPILING)
  add_dependencies(libomp-micro-tests libomp-test-touch)
endif()
if(NOT WIN32 AND NOT APPLE)
  add_dependencies(libomp-micro-tests libomp-test-relo)
endif()
if(NOT WIN32 AND NOT APPLE)
  add_dependencies(libomp-micro-tests libomp-test-execstack)
endif()
if(${MIC})
  add_dependencies(libomp-micro-tests libomp-test-instr)
endif()
add_dependencies(libomp-micro-tests libomp-test-deps)

# Install rules
# We want to install libomp in DESTDIR/CMAKE_INSTALL_PREFIX/lib
# We want to install headers in DESTDIR/CMAKE_INSTALL_PREFIX/include
if(${OPENMP_STANDALONE_BUILD})
  set(LIBOMP_HEADERS_INSTALL_PATH include)
else()
  string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION ${PACKAGE_VERSION})
  set(LIBOMP_HEADERS_INSTALL_PATH "${OPENMP_INSTALL_LIBDIR}/clang/${CLANG_VERSION}/include")
endif()
if(WIN32)
  install(TARGETS omp RUNTIME DESTINATION bin)
  install(TARGETS ompimp ARCHIVE DESTINATION "${OPENMP_INSTALL_LIBDIR}")
  # Create aliases (regular copies) of the library for backwards compatibility
  set(LIBOMP_ALIASES "libiomp5md")
  foreach(alias IN LISTS LIBOMP_ALIASES)
    install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E copy \"${LIBOMP_LIB_FILE}\"
      \"${alias}${LIBOMP_LIBRARY_SUFFIX}\" WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/bin)")
    install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E copy \"${LIBOMP_IMP_LIB_FILE}\"
      \"${alias}${CMAKE_STATIC_LIBRARY_SUFFIX}\" WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/${OPENMP_INSTALL_LIBDIR})")
  endforeach()
else()

  install(TARGETS omp ${LIBOMP_INSTALL_KIND} DESTINATION "${OPENMP_INSTALL_LIBDIR}")

  if(${LIBOMP_INSTALL_ALIASES})
    # Create aliases (symlinks) of the library for backwards compatibility
    set(LIBOMP_ALIASES "libgomp;libiomp5")
    foreach(alias IN LISTS LIBOMP_ALIASES)
      install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \"${LIBOMP_LIB_FILE}\"
        \"${alias}${LIBOMP_LIBRARY_SUFFIX}\" WORKING_DIRECTORY
        \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENMP_INSTALL_LIBDIR})")
    endforeach()
  endif()
endif()
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/omp.h
  DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH}
)
if(${LIBOMP_OMPT_SUPPORT})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/omp-tools.h DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH})
  # install under legacy name ompt.h
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/omp-tools.h DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH} RENAME ompt.h)
endif()
if(${LIBOMP_FORTRAN_MODULES})
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/omp_lib.h
    ${CMAKE_CURRENT_BINARY_DIR}/omp_lib.mod
    ${CMAKE_CURRENT_BINARY_DIR}/omp_lib_kinds.mod
    DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH}
  )
endif()
