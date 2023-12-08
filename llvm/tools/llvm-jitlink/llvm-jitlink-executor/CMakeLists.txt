set(LLVM_LINK_COMPONENTS
  OrcShared
  OrcTargetProcess
  Support
  )

add_llvm_utility(llvm-jitlink-executor
  llvm-jitlink-executor.cpp

  DEPENDS
  intrinsics_gen
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Haiku")
  target_link_libraries(llvm-jitlink-executor PRIVATE network)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "SunOS")
  target_link_libraries(llvm-jitlink-executor PRIVATE socket)
endif()

export_executable_symbols(llvm-jitlink-executor)
