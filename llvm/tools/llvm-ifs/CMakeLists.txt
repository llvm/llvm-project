set(LLVM_LINK_COMPONENTS
  BinaryFormat
  InterfaceStub  
  Object
  Support
  TextAPI
  ObjectYAML
  )

add_llvm_tool(llvm-ifs
  ErrorCollector.cpp
  llvm-ifs.cpp
  )
