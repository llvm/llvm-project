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
  OrcDebugging
  OrcJIT
  OrcShared
  OrcTargetProcess
  RuntimeDyld
  Support
  TargetParser
  )

add_llvm_tool(llvm-jitlink
  llvm-jitlink.cpp
  llvm-jitlink-coff.cpp
  llvm-jitlink-elf.cpp
  llvm-jitlink-macho.cpp
  llvm-jitlink-statistics.cpp

  EXPORT_SYMBOLS
  )

if(${CMAKE_SYSTEM_NAME} MATCHES "Haiku")
  target_link_libraries(llvm-jitlink PRIVATE network)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "SunOS")
  target_link_libraries(llvm-jitlink PRIVATE socket)
endif()

if(WIN32)
  set(builtin_runtime_functions
    llvm_orc_registerEHFrameSectionWrapper
    llvm_orc_deregisterEHFrameSectionWrapper
    llvm_orc_registerJITLoaderGDBWrapper
    llvm_orc_registerJITLoaderGDBAllocAction
    llvm_orc_registerJITLoaderPerfStart
    llvm_orc_registerJITLoaderPerfEnd
    llvm_orc_registerJITLoaderPerfImpl
    llvm_orc_registerVTuneImpl
    llvm_orc_unregisterVTuneImpl
    llvm_orc_test_registerVTuneImpl
    )

  foreach(entry IN LISTS builtin_runtime_functions)
    string(APPEND exports_multiline "${entry}\n")
  endforeach()

  set(file_stem ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/llvm-jitlink)
  file(WRITE ${file_stem}.symbols ${exports_multiline})
  file(WRITE ${file_stem}.def "EXPORTS\n${exports_multiline}")

  add_llvm_symbol_exports(llvm-jitlink ${file_stem}.symbols)
else()
  export_executable_symbols(llvm-jitlink)
endif(WIN32)
