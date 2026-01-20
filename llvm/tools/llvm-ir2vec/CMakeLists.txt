set(LLVM_LINK_COMPONENTS
  # Core LLVM components for IR processing
  Analysis
  Core
  IRReader
  Support
  
  # Machine IR components (for -mode=mir)
  CodeGen           
  MIRParser
  MC
  Target
  
  # Target initialization (required for MIR parsing)
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  TargetParser
  )

# Add the utils subdirectory for the library
add_subdirectory(lib)

# Main executable
add_llvm_tool(llvm-ir2vec
  llvm-ir2vec.cpp
  
  DEPENDS
  intrinsics_gen
)

target_link_libraries(llvm-ir2vec PRIVATE LLVMEmbUtils)
