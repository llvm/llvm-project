set(LLVM_LINK_COMPONENTS
  BinaryFormat
  InterfaceStub
  ObjectYAML
  Option
  Support
  TargetParser
  TextAPI
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(IFSOptsTableGen)

add_llvm_tool(llvm-ifs
  ErrorCollector.cpp
  llvm-ifs.cpp

DEPENDS
  IFSOptsTableGen

  GENERATE_DRIVER
  )

setup_host_tool(llvm-ifs LLVM_IFS llvm_ifs_exe llvm_ifs_target)
