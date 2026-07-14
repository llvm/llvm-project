set(files
  BootstrapInfo.cpp
  SimpleSymbolTable.cpp
  Environment.cpp
  Error.cpp
  ExecutorProcessInfo.cpp
  InProcessControllerAccess.cpp
  Logging.cpp
  NativeDylibManager.cpp
  RTTI.cpp
  Service.cpp
  Session.cpp
  SimpleNativeMemoryMap.cpp
  StandaloneMachOUnwindInfoRegistrar.cpp
  ThreadPoolRunner.cpp
  sps-ci/AllSPSCI.cpp
  sps-ci/CallSPSCI.cpp
  sps-ci/MemoryAccessSPSCI.cpp
  sps-ci/NativeDylibManagerSPSCI.cpp
  sps-ci/SimpleNativeMemoryMapSPSCI.cpp
  sps-ci/StandaloneMachOUnwindInfoRegistrarSPSCI.cpp
  )

# The printf logging backend needs a runtime implementation; the none backend
# is header-only and the os_log backend is not yet implemented.
if(ORC_RT_LOG_BACKEND STREQUAL "printf")
  list(APPEND files Logging_printf.cpp)
elseif(ORC_RT_LOG_BACKEND STREQUAL "os_log")
  list(APPEND files Logging_oslog.cpp)
endif()

add_library(orc-rt-executor STATIC ${files})
target_link_libraries(orc-rt-executor
  PUBLIC orc-rt-headers
  )

# Apply RTTI and exceptions compile flags
# TODO: Use common runtimes infrastructure for output and install paths
target_compile_options(orc-rt-executor PRIVATE ${ORC_RT_COMPILE_FLAGS})
install(TARGETS orc-rt-executor
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
