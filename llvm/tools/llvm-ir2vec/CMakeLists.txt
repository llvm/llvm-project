set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support

  # IR reading (used by llvm-ir2vec.cpp directly)
  IRReader

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

# Build the utility libraries first
add_subdirectory(lib)

# Main executable
add_llvm_tool(llvm-ir2vec
  llvm-ir2vec.cpp

  DEPENDS
  intrinsics_gen
)

# Link both libraries: IR tool for IR mode, MIR tool for MIR mode
target_link_libraries(llvm-ir2vec PRIVATE LLVMIREmbUtils LLVMMIREmbUtils)

option(LLVM_IR2VEC_ENABLE_PYTHON_BINDINGS
  "Build Python bindings for llvm-ir2vec"
  OFF)

set(LLVM_IR2VEC_TEST_PYTHON_BINDINGS False CACHE INTERNAL "")

if(LLVM_IR2VEC_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(Bindings)
  set(LLVM_IR2VEC_TEST_PYTHON_BINDINGS True CACHE INTERNAL "" FORCE)
endif()
