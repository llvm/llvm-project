set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmPrinter
  BitReader
  BitWriter
  CodeGen
  Core
  FuzzMutate
  IRReader
  MC
  ScalarOpts
  SelectionDAG
  Support
  Target
  )

add_llvm_tool(llvm-stress
  llvm-stress.cpp
  mutator.cpp

  DEPENDS
  intrinsics_gen
  )
