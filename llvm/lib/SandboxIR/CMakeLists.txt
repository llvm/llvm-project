add_llvm_component_library(LLVMSandboxIR
  Argument.cpp
  BasicBlock.cpp
  Constant.cpp
  Context.cpp
  Instruction.cpp
  Module.cpp
  Pass.cpp
  PassManager.cpp
  Region.cpp
  Tracker.cpp
  Type.cpp
  User.cpp
  Use.cpp
  Value.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/SandboxIR

  LINK_COMPONENTS
  Core
  Support
  Analysis
  )

