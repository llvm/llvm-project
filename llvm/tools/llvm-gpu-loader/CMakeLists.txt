set(LLVM_LINK_COMPONENTS
  Option
  Object
  BinaryFormat
  TargetParser
  Support
)

add_llvm_tool(llvm-gpu-loader
  llvm-gpu-loader.cpp

  DEPENDS
  intrinsics_gen
)
