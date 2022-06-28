set(LLVM_LINK_COMPONENTS
  Support
  CAS
  )

add_llvm_tool(llvm-cas
  llvm-cas.cpp
  )
