
set(ORC_RT_SOURCES
  BootstrapInfo.cpp
  Environment.cpp
  Error.cpp
  ExecutorProcessInfo.cpp
  GDBJITRegistrar.cpp
  InProcessControllerAccess.cpp
  Logging.cpp
  NativeDylibManager.cpp
  RTTI.cpp
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


# The printf logging backend needs a runtime implementation; the none backend
# is header-only and the os_log backend is not yet implemented.
if(ORC_RT_LOG_BACKEND STREQUAL "printf")
  list(APPEND ORC_RT_SOURCES Logging_printf.cpp)
elseif(ORC_RT_LOG_BACKEND STREQUAL "os_log")
  list(APPEND ORC_RT_SOURCES Logging_oslog.cpp)
endif()

# System implementations are composed, not merely selected: a target gets one
# source list per capability it has. POSIX systems share sys/posix/, and add
# their OS-specific list on top; Windows shares nothing with them.
#
# A file belongs in a shared list only if it is uniform across every member of
# it. Where a function needs an OS conditional, it moves down into the
# per-OS lists instead of growing an #ifdef here -- or, if it wants to inline,
# becomes a dispatch header under include/orc-rt-internal/ (see
# support/sys/CacheControl.h).

set(ORC_RT_POSIX_SOURCES
  sys/posix/DynamicLibrary.cpp
  sys/posix/Errno.cpp
  sys/posix/Memory.cpp
)

set(ORC_RT_DARWIN_SOURCES
  sys/darwin/CPUFeatures.cpp
  sys/darwin/TargetTriple.cpp
)

set(ORC_RT_LINUX_SOURCES
  sys/linux/CPUFeatures.cpp
  sys/linux/TargetTriple.cpp
)

set(ORC_RT_WINDOWS_SOURCES
  sys/windows/CPUFeatures.cpp
  sys/windows/TargetTriple.cpp
)

if (APPLE)
  list(APPEND ORC_RT_SOURCES ${ORC_RT_POSIX_SOURCES} ${ORC_RT_DARWIN_SOURCES})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND ORC_RT_SOURCES ${ORC_RT_POSIX_SOURCES} ${ORC_RT_LINUX_SOURCES})
elseif (WIN32)
  list(APPEND ORC_RT_SOURCES ${ORC_RT_WINDOWS_SOURCES})
else()
  message(WARNING
    "orc-rt: no system implementation for ${CMAKE_SYSTEM_NAME}; "
    "add an orc-rt/lib/bedrock/sys/<system>/ directory and compose it above")
endif()

add_library(orc-rt-bedrock STATIC ${ORC_RT_SOURCES})

target_link_libraries(orc-rt-bedrock PUBLIC orc-rt-headers)

# Apply RTTI and exceptions compile flags
target_compile_options(orc-rt-bedrock PRIVATE ${ORC_RT_COMPILE_FLAGS})
# TODO: Use common runtimes infrastructure for output and install paths
install(TARGETS orc-rt-bedrock
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
