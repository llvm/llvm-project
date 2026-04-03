set(ORC_RT_SOURCES
  AllocAction.cpp
  BootstrapInfo.cpp
  SimpleSymbolTable.cpp
  Error.cpp
  ExecutorProcessInfo.cpp
  QueueingTaskDispatcher.cpp
  RTTI.cpp
  Service.cpp
  Session.cpp
  SimpleNativeMemoryMap.cpp
  TaskDispatcher.cpp
  ThreadPoolTaskDispatcher.cpp
  sps-ci/AllSPSCI.cpp
  sps-ci/SimpleNativeMemoryMapSPSCI.cpp
  )

set(CPU_FEATURE_DETECT_SRC "")

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CPU_FEATURE_DETECT_SRC cpu_feature_detect/darwin/FeatureDetect.cpp)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CPU_FEATURE_DETECT_SRC cpu_feature_detect/linux/FeatureDetect.cpp)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(CPU_FEATURE_DETECT_SRC cpu_feature_detect/windows/FeatureDetect.cpp)
else()
  message(WARNING
        "No CPU feature detection implementation for platform: ${CMAKE_SYSTEM_NAME}. "
        "Add a FeatureDetect.cpp for your platform")
endif()

list(APPEND ORC_RT_SOURCES ${CPU_FEATURE_DETECT_SRC})

add_library(orc-rt-executor STATIC ${ORC_RT_SOURCES})

target_link_libraries(orc-rt-executor
  PUBLIC orc-rt-headers
  )

# Apply RTTI and exceptions compile flags
target_compile_options(orc-rt-executor PRIVATE ${ORC_RT_COMPILE_FLAGS})
install(TARGETS orc-rt-executor
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
