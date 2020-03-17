set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmPrinter
  CodeGen
  Core
  IRReader
  MC
  MIRParser
  Remarks
  ScalarOpts
  SelectionDAG
  Support
  Target
  TransformUtils
  Vectorize
  )

add_llvm_tool(llc
  llc.cpp

  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )

export_executable_symbols(llc)
