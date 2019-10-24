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

# Support plugins.
set(LLVM_SUPPORT_PLUGINS 1)

add_llvm_tool(llc
  llc.cpp

  DEPENDS
  intrinsics_gen
  )
export_executable_symbols(llc)
