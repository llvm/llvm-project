set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BinaryFormat
  ExecutionEngine
  JITLink
  MC
  Object
  OrcJIT
  RuntimeDyld
  Support
  )

add_llvm_tool(llvm-jitlink
  llvm-jitlink.cpp
  llvm-jitlink-macho.cpp
  )

export_executable_symbols(llvm-jitlink)
