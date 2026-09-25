set(LLVM_LINK_COMPONENTS
  BinaryFormat
  DebugInfoCodeView
  DebugInfoDWARF
  Object
  ObjectYAML
  Option
  ProfileData
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(Obj2YamlOptsTableGen)

add_llvm_utility(obj2yaml
  archive2yaml.cpp
  obj2yaml.cpp
  coff2yaml.cpp
  dwarf2yaml.cpp
  dxcontainer2yaml.cpp
  elf2yaml.cpp
  goff2yaml.cpp
  macho2yaml.cpp
  minidump2yaml.cpp
  offload2yaml.cpp
  xcoff2yaml.cpp
  wasm2yaml.cpp
  DEPENDS
  Obj2YamlOptsTableGen
  )
