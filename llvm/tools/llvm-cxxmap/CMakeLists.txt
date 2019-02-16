set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_tool(llvm-cxxmap
  llvm-cxxmap.cpp
  )
