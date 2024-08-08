set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  BitReader
  CodeGen
  Core
  Linker
  LTO
  MC
  Object
  Passes
  Support
  Target
  TargetParser
  )

add_llvm_tool(llvm-lto2
  llvm-lto2.cpp

  DEPENDS
  intrinsics_gen

  EXPORT_SYMBOLS_FOR_PLUGINS
  )
