set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  BitReader
  BitWriter
  CodeGen
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

  DEPENDS intrinsics_gen
  )
