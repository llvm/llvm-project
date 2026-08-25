
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
  sps-ci/AllSPSCI.cpp
  sps-ci/CallSPSCI.cpp
  sps-ci/GDBJITRegistrarSPSCI.cpp
  sps-ci/MemoryAccessSPSCI.cpp
  sps-ci/NativeDylibManagerSPSCI.cpp
  sps-ci/SimpleNativeMemoryMapSPSCI.cpp
  sps-ci/StandaloneMachOUnwindInfoRegistrarSPSCI.cpp
  )


# The printf logging backend needs a runtime implementation; the none backend
# is header-only and the os_log backend is not yet implemented.
if(ORC_RT_LOG_BACKEND STREQUAL "printf")
  list(APPEND ORC_RT_SOURCES Logging_printf.cpp)
elseif(ORC_RT_LOG_BACKEND STREQUAL "os_log")
  list(APPEND ORC_RT_SOURCES Logging_oslog.cpp)
endif()

set(ORC_RT_DARWIN_SOURCES
  darwin/CPUFeatures.cpp
  darwin/TargetTriple.cpp
)

set(ORC_RT_LINUX_SOURCES
  linux/CPUFeatures.cpp
  linux/TargetTriple.cpp
)

set(ORC_RT_WINDOWS_SOURCES
  windows/CPUFeatures.cpp
  windows/TargetTriple.cpp
)

if (APPLE)
    list(APPEND ORC_RT_SOURCES ${ORC_RT_DARWIN_SOURCES})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND ORC_RT_SOURCES ${ORC_RT_LINUX_SOURCES})
  elseif (WIN32)
    list(APPEND ORC_RT_SOURCES ${ORC_RT_WINDOWS_SOURCES})
  else()
    message(WARNING
      "orc-rt: no platform implementation for ${CMAKE_SYSTEM_NAME}; "
      "Add an orc-rt/lib/executor/<platform>/ directory")
endif()

add_library(orc-rt-executor STATIC ${ORC_RT_SOURCES})

target_link_libraries(orc-rt-executor PUBLIC orc-rt-headers)

add_library(orc-rt-executor-impl-headers INTERFACE)
# Add it via lib so that to include impl details its "executor/"
# to prevent collisions down the line
target_include_directories(orc-rt-executor-impl-headers INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

# Apply RTTI and exceptions compile flags
target_compile_options(orc-rt-executor PRIVATE ${ORC_RT_COMPILE_FLAGS})
# TODO: Use common runtimes infrastructure for output and install paths
install(TARGETS orc-rt-executor
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
