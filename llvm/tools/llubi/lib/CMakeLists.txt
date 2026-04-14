set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  )

add_llvm_library(LLVMUBAwareInterpreter
  STATIC
  Context.cpp
  ExecutorBase.cpp
  Interpreter.cpp
  Library.cpp
  Value.cpp
  )
