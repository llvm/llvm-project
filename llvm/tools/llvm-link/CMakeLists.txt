set(LLVM_LINK_COMPONENTS
  BinaryFormat
  BitReader
  BitWriter
  Core
  IRReader
  Linker
  Object
  Option
  TargetParser
  Support
  TransformUtils
  IPO
  )

set(LLVM_TARGET_DEFINITIONS LinkOpts.td)
tablegen(LLVM LinkOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(LinkOptsTableGen)

add_llvm_tool(llvm-link
  llvm-link.cpp

  DEPENDS
  intrinsics_gen
  LinkOptsTableGen
  GENERATE_DRIVER
  )

setup_host_tool(llvm-link LLVM_LINK llvm_link_exe llvm_link_target)
