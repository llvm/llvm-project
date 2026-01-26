set(files
  AllocAction.cpp
  Error.cpp
  QueueingTaskDispatcher.cpp
  ResourceManager.cpp
  RTTI.cpp
  Session.cpp
  SimpleNativeMemoryMap.cpp
  TaskDispatcher.cpp
  ThreadPoolTaskDispatcher.cpp
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
