add_llvm_component_group(Mips HAS_JIT)

set(LLVM_TARGET_DEFINITIONS Mips.td)

llvm_tablegen(MipsGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(MipsGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(MipsGenCallingConv.inc -gen-callingconv)
llvm_tablegen(MipsGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(MipsGenDisassemblerTables.inc -gen-disassembler
              -ignore-non-decodable-operands)
llvm_tablegen(MipsGenFastISel.inc -gen-fast-isel)
llvm_tablegen(MipsGenGlobalISel.inc -gen-global-isel)
llvm_tablegen(MipsGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="MipsPostLegalizerCombiner")
llvm_tablegen(MipsGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(MipsGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(MipsGenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(MipsGenRegisterBank.inc -gen-register-bank)
llvm_tablegen(MipsGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(MipsGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(MipsGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(MipsGenExegesis.inc -gen-exegesis)

set(LLVM_TARGET_DEFINITIONS MipsOptions.td)
llvm_tablegen(MipsOptions.inc -gen-opt-parser-defs)

set(LLVM_TARGET_DEFINITIONS MCTargetDesc/MipsMCOptions.td)
llvm_tablegen(MipsMCOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(MipsCommonTableGen)

add_llvm_target(MipsCodeGen
  Mips16FrameLowering.cpp
  Mips16HardFloat.cpp
  Mips16HardFloatInfo.cpp
  Mips16InstrInfo.cpp
  Mips16ISelDAGToDAG.cpp
  Mips16ISelLowering.cpp
  Mips16RegisterInfo.cpp
  MipsAnalyzeImmediate.cpp
  MipsAsmPrinter.cpp
  MipsCallLowering.cpp
  MipsCCState.cpp
  MipsConstantIslandPass.cpp
  MipsDelaySlotFiller.cpp
  MipsExpandPseudo.cpp
  MipsFastISel.cpp
  MipsInstrInfo.cpp
  MipsInstructionSelector.cpp
  MipsISelDAGToDAG.cpp
  MipsISelLowering.cpp
  MipsFrameLowering.cpp
  MipsLegalizerInfo.cpp
  MipsBranchExpansion.cpp
  MipsMCInstLower.cpp
  MipsMachineFunction.cpp
  MipsOptimizePICCall.cpp
  MipsOs16.cpp
  MipsPreLegalizerCombiner.cpp
  MipsPostLegalizerCombiner.cpp
  MipsRegisterBankInfo.cpp
  MipsRegisterInfo.cpp
  MipsSEFrameLowering.cpp
  MipsSEInstrInfo.cpp
  MipsSEISelDAGToDAG.cpp
  MipsSEISelLowering.cpp
  MipsSERegisterInfo.cpp
  MipsSelectionDAGInfo.cpp
  MipsSubtarget.cpp
  MipsTargetMachine.cpp
  MipsTargetObjectFile.cpp
  MipsTargetTransformInfo.cpp
  MicroMipsSizeReduction.cpp
  MipsMulMulBugPass.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  MipsDesc
  MipsInfo
  Option
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Mips
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
