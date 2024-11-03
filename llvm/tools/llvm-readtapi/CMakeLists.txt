set(LLVM_LINK_COMPONENTS
  Object
  Support
  TextAPI
  )

add_llvm_tool(llvm-readtapi
  llvm-readtapi.cpp
  DiffEngine.cpp
  )
