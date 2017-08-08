set(LLVM_LINK_COMPONENTS
  TransformUtils
  BitWriter
  Core
  IRReader
  Support
  )

add_llvm_tool(llvm-split
  llvm-split.cpp

  DEPENDS
  intrinsics_gen
  )
