add_llvm_library(LLVMGlobalISel
        IRTranslator.cpp
  )

add_dependencies(LLVMGlobalISel intrinsics_gen)
