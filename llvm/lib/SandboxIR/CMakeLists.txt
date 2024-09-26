add_llvm_component_library(LLVMSandboxIR
  Context.cpp
  Module.cpp
  Pass.cpp
  PassManager.cpp
  Region.cpp
  SandboxIR.cpp
  Tracker.cpp
  Type.cpp
  User.cpp
  Value.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/SandboxIR

  LINK_COMPONENTS
  Core
  Support
  Analysis
  )

