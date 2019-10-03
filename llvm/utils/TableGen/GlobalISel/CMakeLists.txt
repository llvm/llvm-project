set(LLVM_LINK_COMPONENTS
  Support
  )

llvm_add_library(LLVMTableGenGlobalISel STATIC
                 CodeExpander.cpp
                )
