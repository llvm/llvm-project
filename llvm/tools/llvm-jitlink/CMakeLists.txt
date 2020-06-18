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
  RuntimeDyld
  Support
  )

add_llvm_tool(llvm-jitlink
  llvm-jitlink.cpp
  llvm-jitlink-elf.cpp
  llvm-jitlink-macho.cpp
  )

export_executable_symbols(llvm-jitlink)
