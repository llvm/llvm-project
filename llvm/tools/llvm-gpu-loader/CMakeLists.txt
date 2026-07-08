set(LLVM_LINK_COMPONENTS
  Option
  Object
  BinaryFormat
  TargetParser
  Support
  FrontendOffloading
)

add_llvm_tool(llvm-gpu-loader
  llvm-gpu-loader.cpp

  DEPENDS
  intrinsics_gen
)
