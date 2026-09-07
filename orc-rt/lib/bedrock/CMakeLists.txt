
set(ORC_RT_BEDROCK_SOURCES
  BootstrapInfo.cpp
  ExecutorProcessInfo.cpp
  GDBJITRegistrar.cpp
  InProcessControllerAccess.cpp
  NativeDylibManager.cpp
  Service.cpp
  Session.cpp
  SimpleNativeMemoryMap.cpp
  SimpleSymbolTable.cpp
  StandaloneMachOUnwindInfoRegistrar.cpp
  ThreadPoolRunner.cpp
  sps/AllSPSCI.cpp
  sps/CallSPSCI.cpp
  sps/GDBJITRegistrarSPSCI.cpp
  sps/MemoryAccessSPSCI.cpp
  sps/NativeDylibManagerSPSCI.cpp
  sps/SimpleNativeMemoryMapSPSCI.cpp
  sps/SimpleRemoteCA.cpp
  sps/StandaloneMachOUnwindInfoRegistrarSPSCI.cpp
  )

# System implementations are composed, not merely selected: a target gets one
# source list per capability it has. POSIX systems share sys/posix/, and add
# their OS-specific list on top; Windows shares nothing with them.
#
# A file belongs in a shared list only if it is uniform across every member of
# it. Where a function needs an OS conditional, it moves down into the
# per-OS lists instead of growing an #ifdef here -- or, if it wants to inline,
# becomes a header with per-system definitions under
# include/orc-rt-internal/support/ (see support/sys/CacheControl.h and
# support/sys/Errno.h).
#
# These live in Bedrock rather than Support because Support is embedded into
# SPIRE, which is JIT-linked on every execution of every JIT'd program. SPIRE
# reaches these services through Bedrock's C API instead of carrying its own
# copy of them.

set(ORC_RT_BEDROCK_POSIX_SOURCES
  sys/posix/DynamicLibrary.cpp
  sys/posix/Memory.cpp
  sys/posix/PageSize.cpp
)

set(ORC_RT_BEDROCK_DARWIN_SOURCES
  sys/darwin/CPUFeatures.cpp
  sys/darwin/TargetTriple.cpp
)

set(ORC_RT_BEDROCK_LINUX_SOURCES
  sys/linux/CPUFeatures.cpp
  sys/linux/TargetTriple.cpp
)

# TODO: Windows has no counterparts for the sys/posix/ sources yet
# (DynamicLibrary, Memory, PageSize), so Windows builds are incomplete.
set(ORC_RT_BEDROCK_WINDOWS_SOURCES
  sys/windows/CPUFeatures.cpp
  sys/windows/TargetTriple.cpp
)

if (APPLE)
  list(APPEND ORC_RT_BEDROCK_SOURCES
    ${ORC_RT_BEDROCK_POSIX_SOURCES} ${ORC_RT_BEDROCK_DARWIN_SOURCES})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND ORC_RT_BEDROCK_SOURCES
    ${ORC_RT_BEDROCK_POSIX_SOURCES} ${ORC_RT_BEDROCK_LINUX_SOURCES})
elseif (WIN32)
  list(APPEND ORC_RT_BEDROCK_SOURCES ${ORC_RT_BEDROCK_WINDOWS_SOURCES})
else()
  message(WARNING
    "orc-rt: no system implementation for ${CMAKE_SYSTEM_NAME}; "
    "add an orc-rt/lib/bedrock/sys/<system>/ directory and compose it above")
endif()

add_library(orc-rt-bedrock STATIC ${ORC_RT_BEDROCK_SOURCES})

target_link_libraries(orc-rt-bedrock PUBLIC orc-rt-headers)

# Support is a build-only implementation detail: its objects are absorbed into
# this archive, and it must not appear in the exported interface.
target_link_libraries(orc-rt-bedrock PRIVATE $<BUILD_INTERFACE:orc-rt-support>)

# Apply RTTI and exceptions compile flags
target_compile_options(orc-rt-bedrock PRIVATE ${ORC_RT_COMPILE_FLAGS})
# TODO: Use common runtimes infrastructure for output and install paths
install(TARGETS orc-rt-bedrock
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
