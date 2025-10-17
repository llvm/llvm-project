set(LLVM_LINK_COMPONENTS
  Option
  Support
  TargetParser
)

add_llvm_tool(llvm-gpu-loader
  llvm-gpu-loader.cpp

  DEPENDS
  intrinsics_gen
)
