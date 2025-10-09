set(LLVM_LINK_COMPONENTS
  Option
  Object
  Support
  TargetParser
)

add_llvm_tool(llvm-gpu-loader
  llvm-gpu-loader.cpp

  DEPENDS
  intrinsics_gen
)
