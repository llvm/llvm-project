add_llvm_component_group(AVR)

set(LLVM_TARGET_DEFINITIONS AVR.td)

llvm_tablegen(AVRGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(AVRGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(AVRGenCallingConv.inc -gen-callingconv)
llvm_tablegen(AVRGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(AVRGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(AVRGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(AVRGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(AVRGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(AVRGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(AVRGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(AVRCommonTableGen)

add_llvm_target(AVRCodeGen
  AVRAsmPrinter.cpp
  AVRCodeGenPassBuilder.cpp
  AVRExpandPseudoInsts.cpp
  AVRFrameLowering.cpp
  AVRInstrInfo.cpp
  AVRISelDAGToDAG.cpp
  AVRISelLowering.cpp
  AVRMCInstLower.cpp
  AVRRegisterInfo.cpp
  AVRSelectionDAGInfo.cpp
  AVRShiftExpand.cpp
  AVRSubtarget.cpp
  AVRTargetMachine.cpp
  AVRTargetObjectFile.cpp
  AVRTargetTransformInfo.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  AVRDesc
  AVRInfo
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  Passes
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  AVR
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
