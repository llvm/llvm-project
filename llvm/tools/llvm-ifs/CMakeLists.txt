set(LLVM_LINK_COMPONENTS
  BinaryFormat
  InterfaceStub  
  Object
  Support
  TextAPI
  ObjectYAML
  Option
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(IFSOptsTableGen)

add_llvm_tool(llvm-ifs
  ErrorCollector.cpp
  llvm-ifs.cpp

DEPENDS
  IFSOptsTableGen
  )
