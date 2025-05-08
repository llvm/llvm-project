# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

add_llvm_component_group(Parasol)

set(LLVM_TARGET_DEFINITIONS Parasol.td)

tablegen(LLVM ParasolGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM ParasolGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM ParasolGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM ParasolGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM ParasolGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM ParasolGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM ParasolGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM ParasolGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM ParasolGenCallingConv.inc -gen-callingconv)
tablegen(LLVM ParasolGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM ParasolGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM ParasolGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM ParasolGenGlobalISel.inc -gen-global-isel)

# ParasolCommonTableGen must be defined
add_public_tablegen_target(ParasolCommonTableGen)

# ParasolCodeGen should match with LLVMBuild.txt ParasolCodeGen
add_llvm_target(ParasolCodeGen
  GISel/ParasolCallLowering.cpp
  GISel/ParasolInstructionSelector.cpp
  GISel/ParasolLegalizerInfo.cpp
  GISel/ParasolRegisterBankInfo.cpp

  AnnotateEncryption.cpp
  ParasolAsmPrinter.cpp
  ParasolFrameLowering.cpp
  ParasolInstrInfo.cpp
  ParasolISelLowering.cpp
  ParasolRegisterInfo.cpp
  ParasolSubtarget.cpp
  ParasolTargetObjectFile.cpp
  ParasolISelDAGToDAG.cpp
  ParasolTargetLowering.cpp
  ParasolTargetMachine.cpp

  # SUNSCREEN TODO: Determine what needs to go into this field.
  LINK_COMPONENTS
  AsmPrinter
  GlobalISel
  MC
  SelectionDAG
  ParasolDesc
  ParasolInfo
  Target

  ADD_TO_COMPONENT
  Parasol
)

# Should match with "subdirectories =  MCTargetDesc TargetInfo" in LLVMBuild.txt
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
