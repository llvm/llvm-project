set(LLVM_LINK_COMPONENTS
  # Core LLVM components for IR processing
  Analysis
  Core
  IRReader
  Support
  
  # Machine IR components (for -mode=mir)
  CodeGen           
  MIRParser         
  
  # Target initialization (required for MIR parsing)
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  TargetParser
  )

add_llvm_tool(llvm-ir2vec
  llvm-ir2vec.cpp
  
  DEPENDS
  intrinsics_gen
  )
