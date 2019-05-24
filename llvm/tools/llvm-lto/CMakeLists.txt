set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  BitReader
  BitWriter
  Core
  IRReader
  LTO
  MC
  Object
  Support
  Target
  )

add_llvm_tool(llvm-lto
  llvm-lto.cpp

  DEPENDS
  intrinsics_gen
  )

