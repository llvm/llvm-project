set(LLVM_LINK_COMPONENTS
  Object
  ObjectYAML
  ProfileData
  Support
  )

add_llvm_utility(yaml2obj
  yaml2obj.cpp
  )
