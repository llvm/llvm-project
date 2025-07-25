if (LLVM_ENABLE_MCCAS)
  add_definitions(-DLLVM_ENABLE_MCCAS=1)
  set(LLVM_LINK_MCCAS
    MCCAS
  )
endif()

set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  CAS
  MC
  ${LLVM_LINK_MCCAS}
  MCParser
  Support
  TargetParser
  DebugInfoDWARF
  DWARFCFIChecker
  )

add_llvm_tool(llvm-mc
  llvm-mc.cpp
  Disassembler.cpp
  )

