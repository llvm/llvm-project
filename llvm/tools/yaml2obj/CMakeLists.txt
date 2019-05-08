set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  MC
  Object
  ObjectYAML
  Support
  )

add_llvm_tool(yaml2obj
  yaml2obj.cpp
  yaml2coff.cpp
  yaml2elf.cpp
  yaml2macho.cpp
  yaml2minidump.cpp
  yaml2wasm.cpp
  )
