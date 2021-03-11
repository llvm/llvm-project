set(LLVM_LINK_COMPONENTS
  InterfaceStub
  Object
  Support
  TextAPI
  ObjectYAML
  )

add_llvm_tool(llvm-ifs
  llvm-ifs.cpp
  )
