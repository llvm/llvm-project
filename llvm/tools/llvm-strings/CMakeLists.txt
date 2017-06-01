set(LLVM_LINK_COMPONENTS
  Core
  Object
  Support
  )

add_llvm_tool(llvm-strings
  llvm-strings.cpp
  )

