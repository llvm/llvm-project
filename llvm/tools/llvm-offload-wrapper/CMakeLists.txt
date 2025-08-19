set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  Object
  Option
  FrontendOffloading
  Support
  TargetParser
  )

add_llvm_tool(llvm-offload-wrapper
  llvm-offload-wrapper.cpp

  DEPENDS
  intrinsics_gen
  )
