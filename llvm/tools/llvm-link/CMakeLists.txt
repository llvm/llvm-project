set(LLVM_LINK_COMPONENTS
  BinaryFormat
  BitReader
  BitWriter
  Core
  IRReader
  Linker
  Object
  Support
  TransformUtils
  IPO
  )

add_llvm_tool(llvm-link
  llvm-link.cpp

  DEPENDS
  intrinsics_gen
  )

setup_host_tool(llvm-link LLVM_LINK llvm_link_exe llvm_link_target)
