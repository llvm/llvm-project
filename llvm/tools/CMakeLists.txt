# NOTE: The tools are organized into groups of four consisting of one large and
# three small executables. This is done to minimize memory load in parallel
# builds.  Please retain this ordering.

if( NOT WIN32 OR MSYS OR CYGWIN )
  # We currently require 'sed' to build llvm-config, so don't try to build it
  # on pure Win32.
  add_subdirectory(llvm-config)
endif()

add_subdirectory(opt)
add_subdirectory(llvm-as)
add_subdirectory(llvm-dis)
add_subdirectory(llvm-mc)

add_subdirectory(llc)
add_subdirectory(llvm-ranlib)
add_subdirectory(llvm-ar)
add_subdirectory(llvm-nm)
add_subdirectory(llvm-size)

add_subdirectory(llvm-cov)
add_subdirectory(llvm-prof)
add_subdirectory(llvm-link)
add_subdirectory(lli)

add_subdirectory(llvm-extract)
add_subdirectory(llvm-diff)
add_subdirectory(macho-dump)
add_subdirectory(llvm-objdump)
add_subdirectory(llvm-readobj)
add_subdirectory(llvm-rtdyld)
add_subdirectory(llvm-dwarfdump)

add_subdirectory(bugpoint)
add_subdirectory(bugpoint-passes)
add_subdirectory(llvm-bcanalyzer)
add_subdirectory(llvm-stress)
add_subdirectory(llvm-mcmarkup)

if( NOT WIN32 )
  add_subdirectory(lto)
endif()

if( LLVM_ENABLE_PIC )
  # TODO: support other systems:
  if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
    add_subdirectory(gold)
  endif()
endif()

add_llvm_external_project(clang)
add_llvm_external_project(lld)
add_llvm_external_project(polly)

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} PARENT_SCOPE)
