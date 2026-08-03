add_llvm_component_group(PISA)

set(LLVM_TARGET_DEFINITIONS PISA.td)
tablegen(LLVM PISAGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM PISAGenGlobalISel.inc -gen-global-isel -warn-on-skipped-patterns -gisel-extended-llt)
tablegen(LLVM PISAGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM PISAGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM PISAGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM PISAGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM PISAGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM PISAGenSearchableTables.inc -gen-searchable-tables)

tablegen(LLVM PISAGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="PISAPreLegalizerCombiner")
tablegen(LLVM PISAGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="PISAPostLegalizerCombiner")

add_public_tablegen_target(PISACommonTableGen)

add_llvm_target(PISACodeGen
  PISACacheHintSelector.cpp
  PISACallLowering.cpp
  PISAConstProp.cpp
  PISAEmitIntrinsics.cpp
  PISAExpandIntrinsics.cpp
  PISAISelLowering.cpp
  PISAInsertLifetimeStart.cpp
  PISAInstrInfo.cpp
  PISAInstructionSelector.cpp
  PISAKernelByValArgsLowering.cpp
  PISALegalizeCalls.cpp
  PISALegalizePredicates.cpp
  PISALegalizeSubregAccess.cpp
  PISALegalizerInfo.cpp
  PISAMCInstLower.cpp
  PISAMachineFunctionInfo.cpp
  PISAMarkConvergentNoMerge.cpp
  PISAOptimizeRedundantCopies.cpp
  PISAOptimizeSubregAccess.cpp
  PISAPostLegalizerCombiner.cpp
  PISAPreLegalizerCombiner.cpp
  PISAPropagateNullPointers.cpp
  PISARegManager.cpp
  PISARegisterBankInfo.cpp
  PISARegisterInfo.cpp
  PISAReplaceIntrinsics.cpp
  PISAScopeSelector.cpp
  PISASubtarget.cpp
  PISATargetMachine.cpp
  PISAUtils.cpp
  PISAVerifier.cpp
  PISAVerifyTypes.cpp
  PISALayout.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  BitReader
  CodeGen
  CodeGenTypes
  Core
  Demangle
  GlobalISel
  IPO
  Linker
  MC
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  PISADesc
  PISAInfo
  Vectorize

  ADD_TO_COMPONENT
  PISA
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
