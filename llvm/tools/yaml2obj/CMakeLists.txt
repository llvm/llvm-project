set(LLVM_LINK_COMPONENTS
  Object
  ObjectYAML
  Option
  ProfileData
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(Yaml2ObjOptsTableGen)

add_llvm_utility(yaml2obj
  yaml2obj.cpp
  DEPENDS
  Yaml2ObjOptsTableGen
  )
