add_llvm_component_group(PowerPC HAS_JIT)

set(LLVM_TARGET_DEFINITIONS PPC.td)

llvm_tablegen(PPCGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(PPCGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(PPCGenCallingConv.inc -gen-callingconv)
llvm_tablegen(PPCGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(PPCGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(PPCGenFastISel.inc -gen-fast-isel)
llvm_tablegen(PPCGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(PPCGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(PPCGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(PPCGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(PPCGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(PPCGenExegesis.inc -gen-exegesis)
llvm_tablegen(PPCGenRegisterBank.inc -gen-register-bank)
llvm_tablegen(PPCGenGlobalISel.inc -gen-global-isel)

add_public_tablegen_target(PowerPCCommonTableGen)

add_llvm_target(PowerPCCodeGen
  GISel/PPCInstructionSelector.cpp
  PPCBoolRetToInt.cpp
  PPCAsmPrinter.cpp
  PPCBranchSelector.cpp
  PPCBranchCoalescing.cpp
  PPCCallingConv.cpp
  PPCCTRLoops.cpp
  PPCCTRLoopsVerify.cpp
  PPCExpandAtomicPseudoInsts.cpp
  PPCHazardRecognizers.cpp
  PPCInstrInfo.cpp
  PPCISelDAGToDAG.cpp
  PPCISelLowering.cpp
  PPCEarlyReturn.cpp
  PPCFastISel.cpp
  PPCFrameLowering.cpp
  PPCLoopInstrFormPrep.cpp
  PPCMCInstLower.cpp
  PPCMachineFunctionInfo.cpp
  PPCMachineScheduler.cpp
  PPCMacroFusion.cpp
  PPCMIPeephole.cpp
  PPCPrepareIFuncsOnAIX.cpp
  PPCRegisterInfo.cpp
  PPCSelectionDAGInfo.cpp
  PPCSubtarget.cpp
  PPCTargetMachine.cpp
  PPCTargetObjectFile.cpp
  PPCTargetTransformInfo.cpp
  PPCTOCRegDeps.cpp
  PPCTLSDynamicCall.cpp
  PPCVSXWACCCopy.cpp
  PPCReduceCRLogicals.cpp
  PPCVSXFMAMutate.cpp
  PPCVSXSwapRemoval.cpp
  PPCPreEmitPeephole.cpp
  PPCLowerMASSVEntries.cpp
  PPCGenScalarMASSEntries.cpp
  GISel/PPCCallLowering.cpp
  GISel/PPCRegisterBankInfo.cpp
  GISel/PPCLegalizerInfo.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  BinaryFormat
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  PowerPCDesc
  PowerPCInfo
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  PowerPC
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
