add_llvm_library(LLVMDFA
    Hello.cpp
    FlowSet.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DataFlowAnalysis

  DEPENDS
  intrinsics_gen
  )
