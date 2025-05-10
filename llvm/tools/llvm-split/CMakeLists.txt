set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  TransformUtils
  BitWriter
  CodeGen
  Core
  FrontendSYCL
  IRReader
  MC
  Support
  Target
  TargetParser
  ipo
  )

add_llvm_tool(llvm-split
  llvm-split.cpp

  DEPENDS
  intrinsics_gen
  )
