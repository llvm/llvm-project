set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_tool(llvm-ctx-ifdo
  llvm-ctx-ifdo.cpp
  )
