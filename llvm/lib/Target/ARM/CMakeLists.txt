add_llvm_component_group(ARM HAS_JIT)

set(LLVM_TARGET_DEFINITIONS ARM.td)

llvm_tablegen(ARMGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(ARMGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(ARMGenCallingConv.inc -gen-callingconv)
llvm_tablegen(ARMGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(ARMGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(ARMGenFastISel.inc -gen-fast-isel)
llvm_tablegen(ARMGenGlobalISel.inc -gen-global-isel)
llvm_tablegen(ARMGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(ARMGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(ARMGenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(ARMGenRegisterBank.inc -gen-register-bank)
llvm_tablegen(ARMGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(ARMGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(ARMGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(ARMGenSystemRegister.inc -gen-searchable-tables)
llvm_tablegen(ARMGenTargetDef.inc -gen-arm-target-def)

add_public_tablegen_target(ARMCommonTableGen)

add_llvm_target(ARMCodeGen
  A15SDOptimizer.cpp
  ARMAsmPrinter.cpp
  ARMBaseInstrInfo.cpp
  ARMBaseRegisterInfo.cpp
  ARMBasicBlockInfo.cpp
  ARMBranchTargets.cpp
  ARMCallingConv.cpp
  ARMCallLowering.cpp
  ARMConstantIslandPass.cpp
  ARMConstantPoolValue.cpp
  ARMExpandPseudoInsts.cpp
  ARMFastISel.cpp
  ARMFixCortexA57AES1742098Pass.cpp
  ARMFrameLowering.cpp
  ARMHazardRecognizer.cpp
  ARMInstructionSelector.cpp
  ARMISelDAGToDAG.cpp
  ARMISelLowering.cpp
  ARMInstrInfo.cpp
  ARMLegalizerInfo.cpp
  ARMParallelDSP.cpp
  ARMLoadStoreOptimizer.cpp
  ARMLowOverheadLoops.cpp
  ARMBlockPlacement.cpp
  ARMMCInstLower.cpp
  ARMMachineFunctionInfo.cpp
  ARMMacroFusion.cpp
  ARMRegisterInfo.cpp
  ARMOptimizeBarriersPass.cpp
  ARMRegisterBankInfo.cpp
  ARMSelectionDAGInfo.cpp
  ARMSLSHardening.cpp
  ARMSubtarget.cpp
  ARMTargetMachine.cpp
  ARMTargetObjectFile.cpp
  ARMTargetTransformInfo.cpp
  MLxExpansionPass.cpp
  MVEGatherScatterLowering.cpp
  MVELaneInterleavingPass.cpp
  MVETailPredication.cpp
  MVEVPTBlockPass.cpp
  MVETPAndVPTOptimisationsPass.cpp
  ARMLatencyMutations.cpp
  Thumb1FrameLowering.cpp
  Thumb1InstrInfo.cpp
  ThumbRegisterInfo.cpp
  Thumb2ITBlockPass.cpp
  Thumb2InstrInfo.cpp
  Thumb2SizeReduction.cpp

  LINK_COMPONENTS
  ARMDesc
  ARMInfo
  ARMUtils
  Analysis
  AsmPrinter
  CFGuard
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  IPO
  MC
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  ARM
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
