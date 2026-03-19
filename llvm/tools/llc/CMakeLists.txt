set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  IPO
  IRPrinter
  IRReader
  MC
  MIRParser
  Option
  Passes
  Plugins
  Remarks
  ScalarOpts
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(LlcOptsTableGen)

add_llvm_tool(llc
  llc.cpp
  NewPMDriver.cpp

  DEPENDS
  LlcOptsTableGen
  intrinsics_gen
  SUPPORT_PLUGINS
  )

export_executable_symbols_for_plugins(llc)
