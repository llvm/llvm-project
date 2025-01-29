#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.20.0)
set(LLVM_SUBPROJECT_TITLE "LLVM libgcc")

set(LLVM_COMMON_CMAKE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# Check if llvm-libgcc is built as a standalone project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR LLVM_LIBGCC_STANDALONE_BUILD)
  project(llvm-libgcc LANGUAGES C CXX ASM)
  set(COMPILER_RT_STANDALONE_BUILD ON)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(LLVM_LIBGCC_COMPILER_RT_BINARY_DIR "compiler-rt")
  set(LLVM_LIBGCC_LIBUNWIND_BINARY_DIR "libunwind")
else()
  set(LLVM_LIBGCC_COMPILER_RT_BINARY_DIR "../compiler-rt")
  set(LLVM_LIBGCC_LIBUNWIND_BINARY_DIR "../libunwind")
endif()

# Add path for custom modules
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/../runtimes/cmake/Modules"
  "${LLVM_COMMON_CMAKE_UTILS}"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
)

set(LLVM_LIBGCC_LIBUNWIND_PATH "${CMAKE_CURRENT_LIST_DIR}/../libunwind"
  CACHE PATH "Specify path to libunwind source.")
set(LLVM_LIBGCC_COMPILER_RT_PATH "${CMAKE_CURRENT_LIST_DIR}/../compiler-rt"
  CACHE PATH "Specify path to compiler-rt source.")

include(GNUInstallDirs)

if(NOT LLVM_LIBGCC_EXPLICIT_OPT_IN)
  message(FATAL_ERROR
    "llvm-libgcc is not for the casual LLVM user. It is intended to be used by distro "
    "managers who want to replace libgcc with compiler-rt and libunwind, but cannot "
    "fully abandon the libgcc family (e.g. because they are dependent on glibc). Such "
    "managers must have worked out their compatibility requirements ahead of using "
    "llvm-libgcc. If you want to build llvm-libgcc, please add -DLLVM_LIBGCC_EXPLICIT_OPT_IN=Yes "
    "to your CMake invocation and try again.")
endif()

if(HAVE_COMPILER_RT)
  message(FATAL_ERROR
    "Attempting to build both compiler-rt and llvm-libgcc will cause irreconcilable "
    "target clashes. Please choose one or the other, but not both.")
endif()

if(HAVE_LIBUNWIND)
  message(FATAL_ERROR
    "Attempting to build both libunwind and llvm-libgcc will cause irreconcilable "
    "target clashes. Please choose one or the other, but not both.")
endif()

#===============================================================================
# Configure System
#===============================================================================

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(TARGET_SUBDIR ${LLVM_DEFAULT_TARGET_TRIPLE})
  if(LLVM_LIBGCC_LIBDIR_SUBDIR)
    string(APPEND TARGET_SUBDIR /${LLVM_LIBGCC_LIBDIR_SUBDIR})
  endif()
  set(LLVM_LIBGCC_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${TARGET_SUBDIR})
  set(LLVM_LIBGCC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${TARGET_SUBDIR} CACHE PATH
      "Path where built llvm-libgcc libraries should be installed.")
  unset(TARGET_SUBDIR)
else()
  if(LLVM_LIBRARY_OUTPUT_INTDIR)
    set(LLVM_LIBGCC_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  else()
    set(LLVM_LIBGCC_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LLVM_LIBGCC_LIBDIR_SUFFIX})
  endif()
  set(LLVM_LIBGCC_INSTALL_LIBRARY_DIR lib${LLVM_LIBGCC_LIBDIR_SUFFIX} CACHE PATH
      "Path where built llvm-libgcc libraries should be installed.")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LLVM_LIBGCC_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LLVM_LIBGCC_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_LIBGCC_LIBRARY_DIR})

#===============================================================================
# Build libraries
#===============================================================================

set(COMPILER_RT_BUILD_BUILTINS ON)
set(COMPILER_RT_BUILTINS_HIDE_SYMBOLS OFF)
add_subdirectory(${LLVM_LIBGCC_COMPILER_RT_PATH} ${LLVM_LIBGCC_COMPILER_RT_BINARY_DIR})

set(LIBUNWIND_ENABLE_STATIC ON)
set(LIBUNWIND_ENABLE_SHARED ON)
set(LIBUNWIND_USE_COMPILER_RT OFF)
set(LIBUNWIND_HAS_GCC_LIB OFF)
set(LIBUNWIND_HAS_GCC_S_LIB OFF)
add_subdirectory(${LLVM_LIBGCC_LIBUNWIND_PATH} ${LLVM_LIBGCC_LIBUNWIND_BINARY_DIR})

add_custom_target(gcc_s.ver
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/gcc_s.ver.in
  COMMAND ${CMAKE_C_COMPILER} -E
    -xc ${CMAKE_CURRENT_SOURCE_DIR}/gcc_s.ver.in
    -o ${CMAKE_CURRENT_BINARY_DIR}/gcc_s.ver
)

add_dependencies(unwind_shared gcc_s.ver)

construct_compiler_rt_default_triple()

target_link_options(unwind_shared PUBLIC
  -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/gcc_s.ver
)

target_link_libraries(unwind_shared PUBLIC
  $<TARGET_OBJECTS:clang_rt.builtins-${COMPILER_RT_DEFAULT_TARGET_ARCH}>
  m
)

#===============================================================================
# Install Symlinks
#===============================================================================

get_compiler_rt_install_dir(${COMPILER_RT_DEFAULT_TARGET_ARCH} install_dir_builtins)
string(REGEX REPLACE "^lib/" "" install_dir_builtins "${install_dir_builtins}")
string(FIND "${install_dir_builtins}" "clang" install_path_contains_triple)
if(install_path_contains_triple EQUAL -1)
  set(builtins_suffix "-${COMPILER_RT_DEFAULT_TARGET_ARCH}")
else()
  string(PREPEND install_dir_builtins "../")
endif()
set(LLVM_LIBGCC_COMPILER_RT ${install_dir_builtins}/libclang_rt.builtins${builtins_suffix}.a)

add_custom_target(llvm-libgcc ALL
  DEPENDS unwind_shared unwind_static clang_rt.builtins-${COMPILER_RT_DEFAULT_TARGET_ARCH}
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${LLVM_LIBGCC_COMPILER_RT} libgcc.a
  COMMAND ${CMAKE_COMMAND} -E create_symlink libunwind.a libgcc_eh.a
  COMMAND ${CMAKE_COMMAND} -E create_symlink libunwind.so libgcc_s.so.1.0
  COMMAND ${CMAKE_COMMAND} -E create_symlink libgcc_s.so.1.0 libgcc_s.so.1
  COMMAND ${CMAKE_COMMAND} -E create_symlink libgcc_s.so.1 libgcc_s.so
)

install(TARGETS unwind_shared unwind_static
  LIBRARY DESTINATION ${LLVM_LIBGCC_INSTALL_LIBRARY_DIR} COMPONENT llvm-libgcc
  ARCHIVE DESTINATION ${LLVM_LIBGCC_INSTALL_LIBRARY_DIR} COMPONENT llvm-libgcc
  RUNTIME DESTINATION ${LLVM_LIBGCC_INSTALL_RUNTIME_DIR} COMPONENT llvm-libgcc)

install(TARGETS clang_rt.builtins-${COMPILER_RT_DEFAULT_TARGET_ARCH}
  LIBRARY DESTINATION ${LLVM_LIBGCC_INSTALL_LIBRARY_DIR}/${install_dir_builtins} COMPONENT llvm-libgcc
  ARCHIVE DESTINATION ${LLVM_LIBGCC_INSTALL_LIBRARY_DIR}/${install_dir_builtins} COMPONENT llvm-libgcc
  RUNTIME DESTINATION ${LLVM_LIBGCC_INSTALL_RUNTIME_DIR}/${install_dir_builtins} COMPONENT llvm-libgcc)

foreach(VAR libgcc.a libgcc_eh.a libgcc_s.so.1.0 libgcc_s.so.1 libgcc_s.so)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${VAR}
    DESTINATION ${LLVM_LIBGCC_INSTALL_LIBRARY_DIR}
    COMPONENT llvm-libgcc)
endforeach()
