add_llvm_component_library(LLVMAtomicCondition
    ACInstrumentation.cpp

    ADDITIONAL_HEADER_DIRS
    ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/ErrorAnalysis/AtomicCondition
    )