set(files
  AllocAction.cpp
  BootstrapInfo.cpp
  SimpleSymbolTable.cpp
  Error.cpp
  ExecutorProcessInfo.cpp
  NativeDylibManager.cpp
  QueueingTaskDispatcher.cpp
  RTTI.cpp
  Service.cpp
  Session.cpp
  SimpleNativeMemoryMap.cpp
  TaskDispatcher.cpp
  ThreadPoolTaskDispatcher.cpp
  sps-ci/AllSPSCI.cpp
  sps-ci/MemoryAccessSPSCI.cpp
  sps-ci/NativeDylibManagerSPSCI.cpp
  sps-ci/SimpleNativeMemoryMapSPSCI.cpp
  )

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
