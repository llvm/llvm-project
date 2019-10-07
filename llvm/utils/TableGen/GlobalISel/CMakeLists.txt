set(LLVM_LINK_COMPONENTS
  Support
  )

llvm_add_library(LLVMTableGenGlobalISel STATIC DISABLE_LLVM_LINK_LLVM_DYLIB
                 CodeExpander.cpp
                )
