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

# Add the lib subdirectory
add_subdirectory(lib)

# Main executable
add_llvm_tool(llvm-ir2vec
  llvm-ir2vec.cpp
  
  DEPENDS
  intrinsics_gen
)

target_link_libraries(llvm-ir2vec PRIVATE LLVMEmbUtils)

option(LLVM_IR2VEC_ENABLE_PYTHON_BINDINGS 
  "Build Python bindings for llvm-ir2vec" 
  OFF)

set(LLVM_IR2VEC_TEST_PYTHON_BINDINGS False CACHE INTERNAL "")

if(LLVM_IR2VEC_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(Bindings)
  set(LLVM_IR2VEC_TEST_PYTHON_BINDINGS True CACHE INTERNAL "" FORCE)
endif()
