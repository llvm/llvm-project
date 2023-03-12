add_llvm_component_library(LLVMUtilities
    FunctionMatchers.cpp
    CompileTimeUtilities.cpp

    ADDITIONAL_HEADER_DIRS
    ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/ErrorAnalysis/Utilities
    )