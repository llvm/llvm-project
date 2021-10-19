set(LLVM_LINK_COMPONENTS
  Object
  Support
  TextAPI
  )

add_llvm_tool(llvm-tapi-diff
  llvm-tapi-diff.cpp
  DiffEngine.cpp
  )
