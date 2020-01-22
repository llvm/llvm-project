# The debugserver build needs to conditionally include files depending on the
# target architecture.
#
# Switch on the architecture specified by TARGET_TRIPLE, as
# the llvm and swift build systems use this variable to identify the
# target (through LLVM_HOST_TRIPLE).
#
# It would be possible to switch on CMAKE_OSX_ARCHITECTURES, but the swift
# build does not provide it, preferring instead to pass arch-specific
# CFLAGS etc explicitly. Switching on LLVM_HOST_TRIPLE is also an option,
# but it breaks down when cross-compiling.

if(TARGET_TRIPLE)
  string(REGEX MATCH "^[^-]*" LLDB_DEBUGSERVER_ARCH ${TARGET_TRIPLE})
else()
  set(LLDB_DEBUGSERVER_ARCH ${CMAKE_OSX_ARCHITECTURES})
endif()

if("${LLDB_DEBUGSERVER_ARCH}" MATCHES ".*arm.*")
  list(APPEND SOURCES arm/DNBArchImpl.cpp arm64/DNBArchImplARM64.cpp)
  include_directories(${CURRENT_SOURCE_DIR}/arm ${CURRENT_SOURCE_DIR}/arm64)
endif()

if(NOT LLDB_DEBUGSERVER_ARCH OR "${LLDB_DEBUGSERVER_ARCH}" MATCHES ".*86.*")
  list(APPEND SOURCES i386/DNBArchImplI386.cpp x86_64/DNBArchImplX86_64.cpp)
  include_directories(${CURRENT_SOURCE_DIR}/i386 ${CURRENT_SOURCE_DIR}/x86_64)
endif()

add_subdirectory(DarwinLog)

include_directories(..)

include_directories(${LLDB_SOURCE_DIR}/tools/debugserver/source)
add_library(lldbDebugserverArchSupport
  ${SOURCES}
  )

set_target_properties(lldbDebugserverArchSupport PROPERTIES FOLDER "lldb libraries/debugserver")
