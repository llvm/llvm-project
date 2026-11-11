add_llvm_component_group(LoongArch)

set(LLVM_TARGET_DEFINITIONS LoongArch.td)

llvm_tablegen(LoongArchGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(LoongArchGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(LoongArchGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(LoongArchGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(LoongArchGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(LoongArchGenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(LoongArchGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(LoongArchGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(LoongArchGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(LoongArchGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(LoongArchCommonTableGen)

add_llvm_target(LoongArchCodeGen
  LoongArchAsmPrinter.cpp
  LoongArchDeadRegisterDefinitions.cpp
  LoongArchExpandAtomicPseudoInsts.cpp
  LoongArchExpandPseudoInsts.cpp
  LoongArchFrameLowering.cpp
  LoongArchInstrInfo.cpp
  LoongArchISelDAGToDAG.cpp
  LoongArchISelLowering.cpp
  LoongArchLateBranchOpt.cpp
  LoongArchMCInstLower.cpp
  LoongArchMemoryBarrierOpt.cpp
  LoongArchMergeBaseOffset.cpp
  LoongArchOptWInstrs.cpp
  LoongArchRegisterInfo.cpp
  LoongArchSelectionDAGInfo.cpp
  LoongArchSubtarget.cpp
  LoongArchTargetMachine.cpp
  LoongArchTargetObjectFile.cpp
  LoongArchTargetTransformInfo.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  LoongArchDesc
  LoongArchInfo
  MC
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  LoongArch
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
