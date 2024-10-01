add_llvm_component_library(LLVMSandboxIR
  Pass.cpp
  PassManager.cpp
  SandboxIR.cpp
  Tracker.cpp
  Type.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/SandboxIR

  LINK_COMPONENTS
  Core
  Support
  Analysis
  )

