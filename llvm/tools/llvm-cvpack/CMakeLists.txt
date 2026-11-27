set(LLVM_LINK_COMPONENTS
  BinaryFormat
  DebugInfoCodeView
  DebugInfoCVPack
  Object
  Support
  )

add_llvm_tool(llvm-cvpack
  llvm-cvpack.cpp
  )
