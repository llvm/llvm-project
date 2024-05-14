add_llvm_component_group(InArch)

set(LLVM_TARGET_DEFINITIONS InArch.td)

tablegen(LLVM InArchGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM InArchGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM InArchGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM InArchGenCallingConv.inc -gen-callingconv)
tablegen(LLVM InArchGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM InArchGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM InArchGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM InArchGenMCCodeEmitter.inc -gen-emitter)

add_public_tablegen_target(InArchCommonTableGen)

add_llvm_target(InArchCodeGen
  InArchTargetMachine.cpp
  InArchRegisterInfo.cpp
  InArchInstrInfo.cpp
  InArchSubtarget.cpp
  InArchAsmPrinter.cpp
  InArchFrameLowering.cpp
  InArchISelDagToDag.cpp
  InArchISelLowering.cpp
  InArchMCInstLower.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  MC
  InArchDesc
  SelectionDAG
  Support
  Target
  InArchInfo
  GlobalISel

  ADD_TO_COMPONENT
  InArch
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)