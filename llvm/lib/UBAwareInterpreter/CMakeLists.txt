set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  )

add_llvm_component_library(LLVMUBAwareInterpreter
  Context.cpp
  ExecutorBase.cpp
  Interpreter.cpp
  Library.cpp
  Value.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/UBAwareInterpreter

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  TargetParser
  )
