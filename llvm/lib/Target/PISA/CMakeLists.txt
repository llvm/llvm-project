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
  PISACallLowering.cpp
  PISAISelLowering.cpp
  PISAInstrInfo.cpp
  PISALegalizerInfo.cpp
  PISAMCInstLower.cpp
  PISAMachineFunctionInfo.cpp
  PISAPostLegalizerCombiner.cpp
  PISAPreLegalizerCombiner.cpp
  PISARegManager.cpp
  PISARegisterBankInfo.cpp
  PISARegisterInfo.cpp
  PISASubtarget.cpp
  PISATargetMachine.cpp
  PISAUtils.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  Demangle
  GlobalISel
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

  ADD_TO_COMPONENT
  PISA
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
