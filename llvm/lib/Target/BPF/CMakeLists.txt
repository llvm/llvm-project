add_llvm_component_group(BPF)

set(LLVM_TARGET_DEFINITIONS BPF.td)

llvm_tablegen(BPFGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(BPFGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(BPFGenCallingConv.inc -gen-callingconv)
llvm_tablegen(BPFGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(BPFGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(BPFGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(BPFGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(BPFGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(BPFGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(BPFGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(BPFGenGlobalISel.inc -gen-global-isel)
llvm_tablegen(BPFGenRegisterBank.inc -gen-register-bank)

add_public_tablegen_target(BPFCommonTableGen)

add_llvm_target(BPFCodeGen
  GISel/BPFCallLowering.cpp
  GISel/BPFInstructionSelector.cpp
  GISel/BPFRegisterBankInfo.cpp
  GISel/BPFLegalizerInfo.cpp
  BPFAbstractMemberAccess.cpp
  BPFAdjustOpt.cpp
  BPFAsmPrinter.cpp
  BPFASpaceCastSimplifyPass.cpp
  BPFCheckAndAdjustIR.cpp
  BPFCodeGenPassBuilder.cpp
  BPFFrameLowering.cpp
  BPFInstrInfo.cpp
  BPFIRPeephole.cpp
  BPFISelDAGToDAG.cpp
  BPFISelLowering.cpp
  BPFMCInstLower.cpp
  BPFPreserveDIType.cpp
  BPFPreserveStaticOffset.cpp
  BPFRegisterInfo.cpp
  BPFSelectionDAGInfo.cpp
  BPFSubtarget.cpp
  BPFTargetLoweringObjectFile.cpp
  BPFTargetMachine.cpp
  BPFUnwind.cpp
  BPFMIPeephole.cpp
  BPFMIChecking.cpp
  BPFMISimplifyPatchable.cpp
  BTFDebug.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  BPFDesc
  BPFInfo
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  IPO
  MC
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  BPF
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
