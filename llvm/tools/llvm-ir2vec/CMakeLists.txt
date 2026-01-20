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

target_include_directories(llvm-ir2vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_link_libraries(llvm-ir2vec PRIVATE LLVMEmbUtils)
