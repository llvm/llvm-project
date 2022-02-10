add_llvm_component_group(LoongArch)

set(LLVM_TARGET_DEFINITIONS LoongArch.td)

tablegen(LLVM LoongArchGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM LoongArchGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM LoongArchGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM LoongArchGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM LoongArchGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM LoongArchGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM LoongArchGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM LoongArchGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(LoongArchCommonTableGen)

add_llvm_target(LoongArchCodeGen
  LoongArchAsmPrinter.cpp
  LoongArchFrameLowering.cpp
  LoongArchInstrInfo.cpp
  LoongArchISelDAGToDAG.cpp
  LoongArchISelLowering.cpp
  LoongArchMCInstLower.cpp
  LoongArchRegisterInfo.cpp
  LoongArchSubtarget.cpp
  LoongArchTargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  MC
  LoongArchDesc
  LoongArchInfo
  SelectionDAG
  Support
  Target
  GlobalISel

  ADD_TO_COMPONENT
  LoongArch
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
