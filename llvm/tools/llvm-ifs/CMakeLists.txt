set(LLVM_LINK_COMPONENTS
  Object
  Support
  TextAPI
  ObjectYAML
  )

add_llvm_tool(llvm-ifs
  llvm-ifs.cpp
  )
