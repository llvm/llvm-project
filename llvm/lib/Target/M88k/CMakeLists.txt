add_llvm_component_group(M88k)

set(LLVM_TARGET_DEFINITIONS M88k.td)

tablegen(LLVM M88kGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM M88kGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM M88kGenCallingConv.inc -gen-callingconv)
tablegen(LLVM M88kGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM M88kGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM M88kGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM M88kGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM M88kGenSubtargetInfo.inc -gen-subtarget)

tablegen(LLVM M88kGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM M88kGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM M88kGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="M88kPreLegalizerCombinerHelper")
tablegen(LLVM M88kGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="M88kPostLegalizerCombinerHelper")
tablegen(LLVM M88kGenPostLegalizeGILowering.inc -gen-global-isel-combiner
              -combiners="M88kPostLegalizerLoweringHelper")

# Generate DAGIsel to identify problems only. It's not used.
tablegen(LLVM M88kGenDAGISel.inc -gen-dag-isel)

add_public_tablegen_target(M88kCommonTableGen)

add_llvm_target(M88kCodeGen
  GISel/M88kCallLowering.cpp
  GISel/M88kInstructionSelector.cpp
  GISel/M88kLegalizerInfo.cpp
  GISel/M88kPreLegalizerCombiner.cpp
  GISel/M88kPostLegalizerLowering.cpp
  GISel/M88kRegisterBankInfo.cpp
  M88kAsmPrinter.cpp
  M88kDelaySlotFiller.cpp
  M88kFrameLowering.cpp
  M88kISelLowering.cpp
  M88kInstrInfo.cpp
#  M88kMachineFunctionInfo.cpp
  M88kMCInstLower.cpp
  M88kRegisterInfo.cpp
  M88kSubtarget.cpp
  M88kTargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  MC
  Scalar
  SelectionDAG
  Support
  M88kDesc
  M88kInfo
  Target
  GlobalISel

  ADD_TO_COMPONENT
  M88k
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
