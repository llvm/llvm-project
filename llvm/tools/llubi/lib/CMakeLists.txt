set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  )

add_llvm_library(LLVMUBAwareInterpreter
  STATIC
  Context.cpp
  Interpreter.cpp
  Value.cpp
  )
