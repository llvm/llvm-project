set(LLVM_LINK_COMPONENTS
  CAS
  Support
  )
add_llvm_tool(llvm-cas-test
  llvm-cas-test.cpp
  )
