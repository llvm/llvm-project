set(LLVM_LINK_COMPONENTS
  # Core LLVM components for IR processing
  Analysis
  Core
  Demangle
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

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_llvm_tool(llvm-ir2vec
  src/llvm-ir2vec.cpp
  src/IR2VecTool.cpp
  src/MIR2VecTool.cpp
  
  DEPENDS
  intrinsics_gen
  )
