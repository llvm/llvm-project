set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  )

add_llvm_component_library(LLVMUBAwareInterpreter
  STATIC
  Context.cpp
  ExecutorBase.cpp
  Interpreter.cpp
  Library.cpp
  Value.cpp
  )
