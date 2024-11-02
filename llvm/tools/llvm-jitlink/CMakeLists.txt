if ( LLVM_INCLUDE_UTILS )
  add_subdirectory(llvm-jitlink-executor)
endif()

set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  BinaryFormat
  ExecutionEngine
  JITLink
  MC
  Object
  OrcJIT
  OrcShared
  OrcTargetProcess
  RuntimeDyld
  Support
  )

add_llvm_tool(llvm-jitlink
  llvm-jitlink.cpp
  llvm-jitlink-coff.cpp
  llvm-jitlink-elf.cpp
  llvm-jitlink-macho.cpp
  )

if(${CMAKE_SYSTEM_NAME} MATCHES "Haiku")
  target_link_libraries(llvm-jitlink PRIVATE network)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "SunOS")
  target_link_libraries(llvm-jitlink PRIVATE socket nsl)
endif()

export_executable_symbols(llvm-jitlink)
