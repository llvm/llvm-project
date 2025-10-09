set(files
  orc-rt-executor.cpp
  AllocAction.cpp
  ResourceManager.cpp
  RTTI.cpp
  Session.cpp
  SimpleNativeMemoryMap.cpp
  )

add_library(orc-rt-executor STATIC ${files})
target_link_libraries(orc-rt-executor
  PUBLIC orc-rt-headers
  )
install(TARGETS orc-rt-executor
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
