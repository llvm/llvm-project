set(LLVM_LINK_COMPONENTS
  AllTargetsInfos
  Object
  Support
  )

add_llvm_tool(llvm-cxxdump
  llvm-cxxdump.cpp
  Error.cpp
  )
