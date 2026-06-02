add_llvm_component_group(EZH)

set(LLVM_TARGET_DEFINITIONS EZH.td)

tablegen(LLVM EZHGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM EZHGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM EZHGenCallingConv.inc -gen-callingconv)
tablegen(LLVM EZHGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM EZHGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM EZHGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM EZHGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM EZHGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM EZHGenSDNodeInfo.inc -gen-sd-node-info)
tablegen(LLVM EZHGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(EZHCommonTableGen)

add_llvm_target(EZHCodeGen
  EZHAsmPrinter.cpp
  EZHBasicBlockInfo.cpp
  EZHBitSliceInjection.cpp
  EZHBranchFixup.cpp
  EZHConstantIslandPass.cpp
  EZHFrameLowering.cpp
  EZHInstrInfo.cpp
  EZHISelDAGToDAG.cpp
  EZHISelLowering.cpp
  EZHMachineFunctionInfo.cpp
  EZHMCInstLower.cpp
  EZHRegisterInfo.cpp
  EZHSelectionDAGInfo.cpp
  EZHSubtarget.cpp
  EZHTargetMachine.cpp
  EZHTargetObjectFile.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  EZHDesc
  EZHInfo
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  EZH
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
