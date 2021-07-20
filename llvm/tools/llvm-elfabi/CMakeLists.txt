set(LLVM_LINK_COMPONENTS
  BinaryFormat
  InterfaceStub
  Object
  Support
  TextAPI
  )

add_llvm_tool(llvm-elfabi
  ErrorCollector.cpp
  llvm-elfabi.cpp
  )
