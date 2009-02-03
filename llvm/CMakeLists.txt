project(LLVM)
cmake_minimum_required(VERSION 2.6.1)

set(PACKAGE_NAME llvm)
set(PACKAGE_VERSION 2.6svn)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed.
CMake would overwrite the makefiles distributed with LLVM.
Please create a directory and run cmake from there, passing the path
to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
endif()

include(FindPerl)

set(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLVM_MAIN_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include)
set(LLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LLVM_TOOLS_BINARY_DIR ${LLVM_BINARY_DIR}/bin)
set(LLVM_EXAMPLES_BINARY_DIR ${LLVM_BINARY_DIR}/examples)

set(LLVM_ALL_TARGETS
  Alpha
  ARM
  CBackend
  CellSPU
  CppBackend
  IA64
  Mips
  MSIL
  PIC16
  PowerPC
  Sparc
  X86
  XCore
  )

# List of targets whose asmprinters need to be forced to link
# into executables on some platforms (i.e. Windows):
set(LLVM_ASMPRINTERS_FORCE_LINK X86 PowerPC)

if( MSVC )
  set(LLVM_TARGETS_TO_BUILD X86
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
else( MSVC )
  set(LLVM_TARGETS_TO_BUILD ${LLVM_ALL_TARGETS}
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
endif( MSVC )

option(LLVM_ENABLE_THREADS "Use threads if available." ON)

if( LLVM_TARGETS_TO_BUILD STREQUAL "all" )
  set( LLVM_TARGETS_TO_BUILD ${LLVM_ALL_TARGETS} )
endif()

foreach(c ${LLVM_TARGETS_TO_BUILD})
  list(FIND LLVM_ALL_TARGETS ${c} idx)
  if( idx LESS 0 )
    message(FATAL_ERROR "The target `${c}' does not exists.
    It should be one of\n${LLVM_ALL_TARGETS}")
  endif()
endforeach(c)

set(llvm_builded_incs_dir ${LLVM_BINARY_DIR}/include/llvm)

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${LLVM_MAIN_SRC_DIR}/cmake"
  "${LLVM_MAIN_SRC_DIR}/cmake/modules"
  )

if(WIN32)
  if(CYGWIN)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
  else(CYGWIN)
    set(LLVM_ON_WIN32 1)
    set(LLVM_ON_UNIX 0)
  endif(CYGWIN)
  set(LTDL_SHLIB_EXT ".dll")
  set(EXEEXT ".exe")
  # Maximum path length is 160 for non-unicode paths
  set(MAXPATHLEN 160)
else(WIN32)
  if(UNIX)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
    set(LTDL_SHLIB_EXT ".so")
    set(EXEEXT "")
    # FIXME: Maximum path length is currently set to 'safe' fixed value
    set(MAXPATHLEN 2024)
  else(UNIX)
    MESSAGE(SEND_ERROR "Unable to determine platform")
  endif(UNIX)
endif(WIN32)

if( EXISTS ${LLVM_TOOLS_BINARY_DIR}/llvm-config )
  set(HAVE_LLVM_CONFIG 1)
endif( EXISTS ${LLVM_TOOLS_BINARY_DIR}/llvm-config )

include(config-ix)

option(LLVM_ENABLE_PIC "Build Position-Independent Code" OFF)

if( LLVM_ENABLE_PIC )
  if( SUPPORTS_FPIC_FLAG )
    message(STATUS "Building with -fPIC")
    add_definitions(-fPIC)
  else( SUPPORTS_FPIC_FLAG )
    message(STATUS "Warning: -fPIC not supported.")
  endif()
endif()

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_TOOLS_BINARY_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib )

# set(CMAKE_VERBOSE_MAKEFILE true)

add_definitions( -D__STDC_LIMIT_MACROS )
add_definitions( -D__STDC_CONSTANT_MACROS )

set(LLVM_PLO_FLAGS "" CACHE
  STRING "Flags for creating partially linked objects.")

if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )
  # TODO: support other platforms and toolchains.
  option(LLVM_BUILD_32_BITS "Build 32 bits executables and libraries." OFF)
  if( LLVM_BUILD_32_BITS )
    message(STATUS "Building 32 bits executables and libraries.")
    add_definitions( -m32 )
    list(APPEND CMAKE_EXE_LINKER_FLAGS -m32)
    list(APPEND CMAKE_SHARED_LINKER_FLAGS -m32)
    set( LLVM_PLO_FLAGS -melf_i386 ${LLVM_PLO_FLAGS} )
  endif( LLVM_BUILD_32_BITS )
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )

if( MSVC )
  add_definitions( -D_CRT_SECURE_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS )
  add_definitions( -D_SCL_SECURE_NO_WARNINGS -DCRT_NONSTDC_NO_WARNINGS )
  add_definitions( -D_SCL_SECURE_NO_DEPRECATE )
  add_definitions( -wd4146 -wd4503 -wd4996 -wd4800 -wd4244 -wd4624 )
  add_definitions( -wd4355 -wd4715 )
endif( MSVC )

include_directories( ${LLVM_BINARY_DIR}/include ${LLVM_MAIN_INCLUDE_DIR})

include(AddLLVM)
include(AddPartiallyLinkedObject)
include(TableGen)

add_subdirectory(lib/Support)
add_subdirectory(lib/System)

# Everything else depends on Support and System:
set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} ${LLVM_LIBS} )

set(LLVM_TABLEGEN "tblgen" CACHE
  STRING "Native TableGen executable. Saves building one when cross-compiling.")

add_subdirectory(utils/TableGen)

if( CMAKE_CROSSCOMPILING )
  # This adds a dependency on target `tblgen', so must go after utils/TableGen
  include( CrossCompileLLVM )
endif( CMAKE_CROSSCOMPILING )

add_subdirectory(include/llvm)

add_subdirectory(lib/VMCore)
add_subdirectory(lib/CodeGen)
add_subdirectory(lib/CodeGen/SelectionDAG)
add_subdirectory(lib/CodeGen/AsmPrinter)
add_subdirectory(lib/Bitcode/Reader)
add_subdirectory(lib/Bitcode/Writer)
add_subdirectory(lib/Transforms/Utils)
add_subdirectory(lib/Transforms/Instrumentation)
add_subdirectory(lib/Transforms/Scalar)
add_subdirectory(lib/Transforms/IPO)
add_subdirectory(lib/Transforms/Hello)
add_subdirectory(lib/Linker)
add_subdirectory(lib/Analysis)
add_subdirectory(lib/Analysis/IPA)

foreach(t ${LLVM_TARGETS_TO_BUILD})
  message(STATUS "Targeting ${t}")
  add_subdirectory(lib/Target/${t})
  if( EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Target/${t}/AsmPrinter/CMakeLists.txt )
    add_subdirectory(lib/Target/${t}/AsmPrinter)
  endif( EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Target/${t}/AsmPrinter/CMakeLists.txt )
endforeach(t)

add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/ExecutionEngine/Interpreter)
add_subdirectory(lib/ExecutionEngine/JIT)
add_subdirectory(lib/Target)
add_subdirectory(lib/AsmParser)
add_subdirectory(lib/Debugger)
add_subdirectory(lib/Archive)

add_subdirectory(tools)

add_subdirectory(examples)

install(DIRECTORY include
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  PATTERN "*.cmake" EXCLUDE
  PATTERN "*.in" EXCLUDE
  )

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include
  DESTINATION .
  )

# TODO: make and install documentation.
