set(LLVM_LINK_COMPONENTS
  ObjectYAML
  Support
  )

add_llvm_tool(yaml2obj
  yaml2obj.cpp
  )
