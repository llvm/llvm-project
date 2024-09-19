#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

include(ExtendPath)

# The generated headers will be placed in clang's resource directory if present.
if(OPENMP_STANDALONE_BUILD OR NOT LLVM_RUNTIMES_BUILD)
  set(LIBOMP_HEADERS_INTDIR ${CMAKE_CURRENT_BINARY_DIR})
else()
  set(LIBOMP_HEADERS_INTDIR ${LLVM_BINARY_DIR}/${LIBOMP_HEADERS_INSTALL_PATH})
endif()

# Configure omp.h, kmp_config.h and omp-tools.h if necessary
configure_file(${LIBOMP_INC_DIR}/omp.h.var ${LIBOMP_HEADERS_INTDIR}/omp.h @ONLY)
configure_file(${LIBOMP_INC_DIR}/ompx.h.var ${LIBOMP_HEADERS_INTDIR}/ompx.h @ONLY)
configure_file(kmp_config.h.cmake kmp_config.h @ONLY)
if(${LIBOMP_OMPT_SUPPORT})
  configure_file(${LIBOMP_INC_DIR}/omp-tools.h.var ${LIBOMP_HEADERS_INTDIR}/omp-tools.h @ONLY)
endif()

# Generate message catalog files: kmp_i18n_id.inc and kmp_i18n_default.inc
add_custom_command(
  OUTPUT  kmp_i18n_id.inc
  COMMAND ${Python3_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/message-converter.py
          --enum=kmp_i18n_id.inc ${LIBOMP_SRC_DIR}/i18n/en_US.txt
  DEPENDS ${LIBOMP_SRC_DIR}/i18n/en_US.txt ${LIBOMP_TOOLS_DIR}/message-converter.py
)
add_custom_command(
  OUTPUT  kmp_i18n_default.inc
  COMMAND ${Python3_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/message-converter.py
          --default=kmp_i18n_default.inc ${LIBOMP_SRC_DIR}/i18n/en_US.txt
  DEPENDS ${LIBOMP_SRC_DIR}/i18n/en_US.txt ${LIBOMP_TOOLS_DIR}/message-converter.py
)

# Set the -D definitions for all sources
# UNICODE and _UNICODE are set in LLVM's CMake system.  They affect the
# ittnotify code and should only be set when compiling ittnotify_static.cpp
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

# Building with time profiling support requires LLVM directory includes.
if(LIBOMP_PROFILING_SUPPORT)
  include_directories(
    ${LLVM_MAIN_INCLUDE_DIR}
    ${LLVM_INCLUDE_DIR}
  )
endif()

# Getting correct source files to build library
set(LIBOMP_CXXFILES)
set(LIBOMP_ASMFILES)
set(LIBOMP_GNUASMFILES)
if(STUBS_LIBRARY)
  set(LIBOMP_CXXFILES kmp_stub.cpp)
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
    kmp_collapse.cpp
  )
  if(WIN32)
    # Windows specific files
    libomp_append(LIBOMP_CXXFILES z_Windows_NT_util.cpp)
    libomp_append(LIBOMP_CXXFILES z_Windows_NT-586_util.cpp)
    if(${LIBOMP_ARCH} STREQUAL "i386" OR ${LIBOMP_ARCH} STREQUAL "x86_64")
      libomp_append(LIBOMP_ASMFILES z_Windows_NT-586_asm.asm) # Windows assembly file
    elseif((${LIBOMP_ARCH} STREQUAL "aarch64" OR ${LIBOMP_ARCH} STREQUAL "arm") AND (NOT MSVC OR CMAKE_C_COMPILER_ID STREQUAL "Clang"))
      # z_Linux_asm.S works for AArch64 and ARM Windows too.
      libomp_append(LIBOMP_GNUASMFILES z_Linux_asm.S)
    else()
      # AArch64 with MSVC gets implementations of the functions from
      # ifdeffed sections in z_Windows_NT-586_util.cpp.
    endif()
  else()
    # Unix specific files
    libomp_append(LIBOMP_CXXFILES z_Linux_util.cpp)
    libomp_append(LIBOMP_CXXFILES kmp_gsupport.cpp)
    if(${CMAKE_SYSTEM_NAME} MATCHES "AIX")
      libomp_append(LIBOMP_GNUASMFILES z_AIX_asm.S) # AIX assembly file
    else()
      libomp_append(LIBOMP_GNUASMFILES z_Linux_asm.S) # Unix assembly file
    endif()
  endif()
  libomp_append(LIBOMP_CXXFILES thirdparty/ittnotify/ittnotify_static.cpp LIBOMP_USE_ITT_NOTIFY)
  libomp_append(LIBOMP_CXXFILES kmp_debugger.cpp LIBOMP_USE_DEBUGGER)
  libomp_append(LIBOMP_CXXFILES kmp_stats.cpp LIBOMP_STATS)
  libomp_append(LIBOMP_CXXFILES kmp_stats_timing.cpp LIBOMP_STATS)
  libomp_append(LIBOMP_CXXFILES kmp_taskdeps.cpp)
  libomp_append(LIBOMP_CXXFILES kmp_cancel.cpp)
endif()
# Files common to stubs and normal library
libomp_append(LIBOMP_CXXFILES kmp_ftn_cdecl.cpp)
libomp_append(LIBOMP_CXXFILES kmp_ftn_extra.cpp)
libomp_append(LIBOMP_CXXFILES kmp_version.cpp)
libomp_append(LIBOMP_CXXFILES ompt-general.cpp IF_TRUE LIBOMP_OMPT_SUPPORT)
libomp_append(LIBOMP_CXXFILES ompd-specific.cpp IF_TRUE LIBOMP_OMPD_SUPPORT)

set(LIBOMP_SOURCE_FILES ${LIBOMP_CXXFILES} ${LIBOMP_ASMFILES} ${LIBOMP_GNUASMFILES})
# For Windows, there is a resource file (.rc -> .res) that is also compiled
libomp_append(LIBOMP_SOURCE_FILES libomp.rc WIN32)

# Get compiler and assembler flags
libomp_get_cxxflags(LIBOMP_CONFIGURED_CXXFLAGS)
libomp_get_asmflags(LIBOMP_CONFIGURED_ASMFLAGS)
# Set the compiler flags for each type of source
set_source_files_properties(${LIBOMP_CXXFILES} PROPERTIES COMPILE_FLAGS "${LIBOMP_CONFIGURED_CXXFLAGS}")
set_source_files_properties(${LIBOMP_ASMFILES} ${LIBOMP_GNUASMFILES} PROPERTIES COMPILE_FLAGS "${LIBOMP_CONFIGURED_ASMFLAGS}")

# Remove any cmake-automatic linking of the standard C++ library.
# We neither need (nor want) the standard C++ library dependency even though we compile c++ files.
if(NOT ${LIBOMP_USE_STDCPPLIB})
  set(LIBOMP_LINKER_LANGUAGE C)
  set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES)
else()
  set(LIBOMP_LINKER_LANGUAGE CXX)
endif()

if(UNIX)
  set(LIBOMP_DL_LIBS ${CMAKE_DL_LIBS})
endif()

# Disable libstdc++ assertions, even in an LLVM_ENABLE_ASSERTIONS build, to
# avoid an unwanted dependency on libstdc++.so.
if(NOT WIN32)
  add_definitions(-U_GLIBCXX_ASSERTIONS)
endif()

# Add the OpenMP library
libomp_get_ldflags(LIBOMP_CONFIGURED_LDFLAGS)

libomp_get_libflags(LIBOMP_CONFIGURED_LIBFLAGS)
# Build libomp library. Add LLVMSupport dependency if building in-tree with libomptarget profiling enabled.
if(OPENMP_STANDALONE_BUILD OR (NOT OPENMP_ENABLE_LIBOMP_PROFILING))
  add_library(omp ${LIBOMP_LIBRARY_KIND} ${LIBOMP_SOURCE_FILES})
  set_property(TARGET omp PROPERTY FOLDER "OpenMP/Libraries")
  # Linking command will include libraries in LIBOMP_CONFIGURED_LIBFLAGS
  target_link_libraries(omp ${LIBOMP_CONFIGURED_LIBFLAGS} ${LIBOMP_DL_LIBS})
else()
  add_llvm_library(omp ${LIBOMP_LIBRARY_KIND} ${LIBOMP_SOURCE_FILES} PARTIAL_SOURCES_INTENDED
    LINK_LIBS ${LIBOMP_CONFIGURED_LIBFLAGS} ${LIBOMP_DL_LIBS}
    LINK_COMPONENTS Support
    BUILDTREE_ONLY
    )
  # libomp must be a C++ library such that it can link libLLVMSupport
  set(LIBOMP_LINKER_LANGUAGE CXX)
endif()
if(${LIBOMP_USE_HWLOC})
  target_include_directories(omp
                             PUBLIC
			     "$<BUILD_INTERFACE:${LIBOMP_HWLOC_INCLUDE_DIR}>"
			     "$<INSTALL_INTERFACE:${LIBOMP_HWLOC_INCLUDE_DIR}>"
  )
endif()

if(OPENMP_MSVC_NAME_SCHEME)
  if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    set(LIBOMP_PDB_NAME ${LIBOMP_DEFAULT_LIB_NAME}${MSVC_TOOLS_VERSION}d.${LIBOMP_ARCH})
    set(LIBOMP_LIB_FILE ${LIBOMP_PDB_NAME}${LIBOMP_LIBRARY_SUFFIX})
  else()
    # ${LIBOMP_LIB_NAME} is ${LIBOMP_DEFAULT_LIB_NAME}${MSVC_TOOLS_VERSION}.${LIBOMP_ARCH}
    set(LIBOMP_PDB_NAME ${LIBOMP_LIB_NAME})
  endif()

  # in debug
  # LIBOMP_LIB_FILE should be LIBOMP_LIB_FILE_DBG = ${LIBOMP_LIB_NAME_DBG}${LIBOMP_LIBRARY_SUFFIX}
  #						= ${LIBOMP_DEFAULT_LIB_NAME}${MSVC_TOOLS_VERSION}d.${LIBOMP_ARCH}${LIBOMP_LIBRARY_SUFFIX}
  # COMPILE_PDB_NAME/PDB_NAME should be LIBOMP_LIB_NAME_DBG = ${LIBOMP_DEFAULT_LIB_NAME}${MSVC_TOOLS_VERSION}d.${LIBOMP_ARCH}
  set_target_properties(omp PROPERTIES
    PREFIX "" SUFFIX "" OUTPUT_NAME "${LIBOMP_LIB_FILE}"
    PDB_NAME "${LIBOMP_PDB_NAME}"
    COMPILE_PDB_NAME "${LIBOMP_PDB_NAME}"
    LINK_FLAGS "${LIBOMP_CONFIGURED_LDFLAGS}"
    LINKER_LANGUAGE ${LIBOMP_LINKER_LANGUAGE}
 )
else()
  set_target_properties(omp PROPERTIES
    PREFIX "" SUFFIX "" OUTPUT_NAME "${LIBOMP_LIB_FILE}"
    LINK_FLAGS "${LIBOMP_CONFIGURED_LDFLAGS}"
    LINKER_LANGUAGE ${LIBOMP_LINKER_LANGUAGE}
  )
endif()

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
set(LIBOMP_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBOMP_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

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

# Definitions for testing, for reuse when testing libomptarget-nvptx.
set(LIBOMPTARGET_OPENMP_HEADER_FOLDER "${LIBOMP_INCLUDE_DIR}" CACHE STRING
  "Path to folder containing omp.h")
set(LIBOMPTARGET_OPENMP_HOST_RTL_FOLDER "${LIBOMP_LIBRARY_DIR}" CACHE STRING
  "Path to folder containing libomp.so, and libLLVMSupport.so with profiling enabled")

# Create *.inc before compiling any sources
# objects depend on : .inc files
add_custom_target(libomp-needed-headers DEPENDS kmp_i18n_id.inc kmp_i18n_default.inc)
set_target_properties(libomp-needed-headers PROPERTIES FOLDER "OpenMP/Sourcegenning")
add_dependencies(omp libomp-needed-headers)

# Windows specific build rules
if(WIN32)
  configure_file(libomp.rc.var libomp.rc @ONLY)
  # z_Windows_NT-586_asm.asm requires definitions to be sent via command line
  # It only needs the architecture macro and OMPT_SUPPORT=0|1
  libomp_append(LIBOMP_MASM_DEFINITIONS "-D_M_IA32" IF_TRUE IA32)
  libomp_append(LIBOMP_MASM_DEFINITIONS "-D_M_AMD64" IF_TRUE INTEL64)
  libomp_append(LIBOMP_MASM_DEFINITIONS "-DOMPT_SUPPORT" IF_TRUE_1_0 LIBOMP_OMPT_SUPPORT)
  libomp_append(LIBOMP_MASM_DEFINITIONS "-DOMPD_SUPPORT" IF_TRUE_1_0 LIBOMP_OMPD_SUPPORT)
  libomp_list_to_string("${LIBOMP_MASM_DEFINITIONS}" LIBOMP_MASM_DEFINITIONS)
  set_property(SOURCE z_Windows_NT-586_asm.asm APPEND_STRING PROPERTY COMPILE_FLAGS " ${LIBOMP_MASM_DEFINITIONS}")
  set_source_files_properties(thirdparty/ittnotify/ittnotify_static.cpp PROPERTIES COMPILE_DEFINITIONS "UNICODE")

  # Create Windows import library
  # On debug LIBOMP_IMP_LIB_FILE should be LIBOMP_IMP_LIB_FILE_DBG = ${LIBOMP_DEFAULT_LIB_NAME_DBG}${CMAKE_STATIC_LIBRARY_SUFFIX}
  # 									${LIBOMP_DEFAULT_LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX}
  # and the ARCHIVE_OUTPUT_NAME of ompdbg should be ${LIBOMP_DEFAULT_LIB_NAME_DBG}${LIBOMP_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}

  if(OPENMP_MSVC_NAME_SCHEME)
    if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
      set(LIBOMP_IMP_LIB_FILE ${LIBOMP_DEFAULT_LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(LIBOMP_GENERATED_IMP_LIB_FILENAME ${LIBOMP_DEFAULT_LIB_NAME}d${LIBOMP_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      set(LIBOMP_IMP_LIB_FILE ${LIBOMP_DEFAULT_LIB_NAME}${CMAKE_IMPORT_LIBRARY_SUFFIX})
      set(LIBOMP_GENERATED_IMP_LIB_FILENAME ${LIBOMP_DEFAULT_LIB_NAME}${LIBOMP_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
  else()
    set(LIBOMP_IMP_LIB_FILE ${LIBOMP_LIB_NAME}${CMAKE_IMPORT_LIBRARY_SUFFIX})
    set(LIBOMP_GENERATED_IMP_LIB_FILENAME ${LIBOMP_LIB_FILE}${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
  set_target_properties(omp PROPERTIES
    VERSION ${LIBOMP_VERSION_MAJOR}.${LIBOMP_VERSION_MINOR} # uses /version flag
    IMPORT_PREFIX "" IMPORT_SUFFIX "" # control generated import library name when building omp
    ARCHIVE_OUTPUT_NAME ${LIBOMP_GENERATED_IMP_LIB_FILENAME}
  )

  set(LIBOMP_IMP_LIB_TARGET omp)
  set(LIBOMP_GENERATED_DEF_FILE ${LIBOMP_LIB_NAME}.def)
  add_custom_target(libomp-needed-def-file DEPENDS ${LIBOMP_GENERATED_DEF_FILE})
  set_target_properties(libomp-needed-def-file PROPERTIES FOLDER "OpenMP/Sourcegenning")
  add_dependencies(omp libomp-needed-def-file)

  # Create the main def file with ordinals to use for building the runtime dll to maintain backwards compatible exports order
  libomp_get_gdflags(LIBOMP_GDFLAGS)
  libomp_string_to_list("${LIBOMP_GDFLAGS}" LIBOMP_GDFLAGS)

  add_custom_command(
    OUTPUT  ${LIBOMP_GENERATED_DEF_FILE}
    COMMAND ${Python3_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/generate-def.py ${LIBOMP_GDFLAGS} --name ${LIBOMP_LIB_FILE}
            -o ${LIBOMP_GENERATED_DEF_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/dllexports
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dllexports ${LIBOMP_TOOLS_DIR}/generate-def.py
  )

  if(MSVC)
    # For toolchains that generated the import library importing by ordinal, re-generate it to import by name
    set(LIBOMP_IMP_LIB_TARGET ompimp)
    # Create the auxiliary def file without ordinals to use for building the import library to import by name
    set(LIBOMPIMP_GENERATED_DEF_FILE ${LIBOMP_LIB_NAME}.imp.def)
    add_custom_target(libompimp-needed-def-file DEPENDS ${LIBOMPIMP_GENERATED_DEF_FILE})
    set_target_properties(libompimp-needed-def-file PROPERTIES FOLDER "OpenMP/Resources")
    add_custom_command(
      OUTPUT  ${LIBOMPIMP_GENERATED_DEF_FILE}
      COMMAND ${Python3_EXECUTABLE} ${LIBOMP_TOOLS_DIR}/generate-def.py ${LIBOMP_GDFLAGS}
              --name ${LIBOMP_LIB_FILE} --no-ordinals
              -o ${LIBOMPIMP_GENERATED_DEF_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/dllexports
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dllexports ${LIBOMP_TOOLS_DIR}/generate-def.py
    )
    # while this merely generates an import library off a def file, CMAKE still requires it to have a "source" so feed it a dummy one,
    # making it a .txt which CMAKE will filter out from the librarian (a .cpp will make lib.exe punt trying to resolve the .def symbols)
    add_library(${LIBOMP_IMP_LIB_TARGET} STATIC kmp_dummy.txt)
    set_target_properties(${LIBOMP_IMP_LIB_TARGET} PROPERTIES FOLDER "OpenMP/Libraries")
    set_target_properties(${LIBOMP_IMP_LIB_TARGET} PROPERTIES
        PREFIX "" SUFFIX "" OUTPUT_NAME "${LIBOMP_IMP_LIB_FILE}" LINKER_LANGUAGE ${LIBOMP_LINKER_LANGUAGE}
        STATIC_LIBRARY_OPTIONS "${CMAKE_LINK_DEF_FILE_FLAG}${CMAKE_CURRENT_BINARY_DIR}/${LIBOMPIMP_GENERATED_DEF_FILE}")
    add_dependencies(${LIBOMP_IMP_LIB_TARGET} libompimp-needed-def-file)
    add_dependencies(omp ${LIBOMP_IMP_LIB_TARGET})
  endif()
endif()

# Building the Fortran module files
# One compilation step creates both omp_lib.mod and omp_lib_kinds.mod
configure_file(${LIBOMP_INC_DIR}/omp_lib.h.var omp_lib.h @ONLY)
configure_file(${LIBOMP_INC_DIR}/omp_lib.F90.var omp_lib.F90 @ONLY)

set(BUILD_FORTRAN_MODULES False)
if (NOT ${LIBOMP_FORTRAN_MODULES_COMPILER} STREQUAL "")
  # If libomp is built as an LLVM runtime and the flang compiler is available,
  # compile the Fortran module files.
  message(STATUS "configuring openmp to build Fortran module files using ${LIBOMP_FORTRAN_MODULES_COMPILER}")
  set(LIBOMP_FORTRAN_SOURCE_FILE omp_lib.F90)
  add_custom_target(libomp-mod ALL DEPENDS omp_lib.mod omp_lib_kinds.mod)
  add_custom_command(
    OUTPUT omp_lib.mod omp_lib_kinds.mod
    COMMAND ${LIBOMP_FORTRAN_MODULES_COMPILER} -cpp -fsyntax-only ${LIBOMP_FORTRAN_SOURCE_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LIBOMP_FORTRAN_SOURCE_FILE}
      ${CMAKE_CURRENT_BINARY_DIR}/omp_lib.h
  )
  set(BUILD_FORTRAN_MODULES True)
elseif(${LIBOMP_FORTRAN_MODULES})
  # The following requests explicit building of the Fortran module files
  # Workaround for gfortran to build modules with the
  # omp_sched_monotonic integer parameter
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(ADDITIONAL_Fortran_FLAGS "-fno-range-check")
  endif()
  add_custom_target(libomp-mod ALL DEPENDS omp_lib.mod omp_lib_kinds.mod)
  set_target_properties(libomp-mod PROPERTIES FOLDER "OpenMP/Misc")
  libomp_get_fflags(LIBOMP_CONFIGURED_FFLAGS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    set(LIBOMP_FORTRAN_SOURCE_FILE omp_lib.F90)
  else()
    message(FATAL_ERROR "Fortran module build requires Fortran 90 compiler")
  endif()
  add_custom_command(
    OUTPUT omp_lib.mod omp_lib_kinds.mod
    COMMAND ${CMAKE_Fortran_COMPILER} -c ${ADDITIONAL_Fortran_FLAGS}
            ${LIBOMP_CONFIGURED_FFLAGS} ${LIBOMP_FORTRAN_SOURCE_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LIBOMP_FORTRAN_SOURCE_FILE}
      ${CMAKE_CURRENT_BINARY_DIR}/omp_lib.h
  )
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES omp_lib${CMAKE_C_OUTPUT_EXTENSION})
  set(BUILD_FORTRAN_MODULES True)
endif()

# Move files to exports/ directory if requested
if(${LIBOMP_COPY_EXPORTS})
  include(LibompExports)
endif()

# Micro test rules for after library has been built (cmake/LibompMicroTests.cmake)
include(LibompMicroTests)
add_custom_target(libomp-micro-tests)
set_target_properties(libomp-micro-tests PROPERTIES FOLDER "OpenMP/Tests")
if(NOT ${MIC} AND NOT CMAKE_CROSSCOMPILING)
  add_dependencies(libomp-micro-tests libomp-test-touch)
endif()
if(NOT WIN32 AND NOT APPLE)
  add_dependencies(libomp-micro-tests libomp-test-relo)
endif()
if(NOT WIN32 AND NOT APPLE)
  add_dependencies(libomp-micro-tests libomp-test-execstack)
endif()
add_dependencies(libomp-micro-tests libomp-test-deps)

# `omp` needs to be exported if in-tree build.
set(export_to_llvmexports)
if (NOT OPENMP_STANDALONE_BUILD)
  get_target_export_arg(omp LLVM export_to_llvmexports)
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS omp)
endif()

# Install rules
# We want to install libomp in ${DESTDIR}/${CMAKE_INSTALL_FULL_LIBDIR}
# We want to install headers in ${DESTDIR}/${CMAKE_INSTALL_FULL_INCLUDEDIR}
if(WIN32)
  install(TARGETS omp ${export_to_llvmexports} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(TARGETS ${LIBOMP_IMP_LIB_TARGET} ARCHIVE DESTINATION "${OPENMP_INSTALL_LIBDIR}")
  # Create aliases (regular copies) of the library for backwards compatibility
  set(LIBOMP_ALIASES "libiomp5md")
  foreach(alias IN LISTS LIBOMP_ALIASES)
    install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E copy \"${LIBOMP_LIB_FILE}\"
      \"${alias}${LIBOMP_LIBRARY_SUFFIX}\" WORKING_DIRECTORY \"${CMAKE_INSTALL_FULL_BINDIR}\")")
    extend_path(outdir "${CMAKE_INSTALL_PREFIX}" "${OPENMP_INSTALL_LIBDIR}")
    install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E copy \"${LIBOMP_IMP_LIB_FILE}\"
      \"${alias}${CMAKE_STATIC_LIBRARY_SUFFIX}\" WORKING_DIRECTORY \"${outdir}\")")
  endforeach()
else()

  if(${CMAKE_SYSTEM_NAME} MATCHES "AIX")
    install(FILES ${LIBOMP_LIBRARY_DIR}/libomp.a DESTINATION "${OPENMP_INSTALL_LIBDIR}" COMPONENT runtime)
  else()
    install(TARGETS omp ${export_to_llvmexports} ${LIBOMP_INSTALL_KIND} DESTINATION "${OPENMP_INSTALL_LIBDIR}")
  endif()

  if(${LIBOMP_INSTALL_ALIASES})
    # Create aliases (symlinks) of the library for backwards compatibility
    set(LIBOMP_ALIASES "libgomp;libiomp5")
    foreach(alias IN LISTS LIBOMP_ALIASES)
      extend_path(outdir "${CMAKE_INSTALL_PREFIX}" "${OPENMP_INSTALL_LIBDIR}")
      install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \"${LIBOMP_LIB_FILE}\"
        \"${alias}${LIBOMP_LIBRARY_SUFFIX}\" WORKING_DIRECTORY
        \"\$ENV{DESTDIR}${outdir}\")")
    endforeach()
  endif()
endif()
install(
  FILES
  ${LIBOMP_HEADERS_INTDIR}/omp.h
  ${LIBOMP_HEADERS_INTDIR}/ompx.h
  DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH}
)
if(${LIBOMP_OMPT_SUPPORT})
  install(FILES ${LIBOMP_HEADERS_INTDIR}/omp-tools.h DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH})
  # install under legacy name ompt.h
  install(FILES ${LIBOMP_HEADERS_INTDIR}/omp-tools.h DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH} RENAME ompt.h)
  set(LIBOMP_OMP_TOOLS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
endif()
if(${BUILD_FORTRAN_MODULES})
  set (destination ${LIBOMP_HEADERS_INSTALL_PATH})
  if (NOT ${LIBOMP_MODULES_INSTALL_PATH} STREQUAL "")
    set (destination ${LIBOMP_MODULES_INSTALL_PATH})
  endif()
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/omp_lib.h
    ${CMAKE_CURRENT_BINARY_DIR}/omp_lib.mod
    ${CMAKE_CURRENT_BINARY_DIR}/omp_lib_kinds.mod
    DESTINATION ${destination}
  )
endif()
