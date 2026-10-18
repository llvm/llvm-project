add_llvm_component_group(AGX)

set(LLVM_TARGET_DEFINITIONS AGX.td)

tablegen(LLVM AGXGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AGXGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AGXGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AGXGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AGXGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM AGXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AGXGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM AGXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AGXGenSDNodeInfo.inc -gen-sd-node-info)
tablegen(LLVM AGXGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM AGXGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(AGXCommonTableGen)

add_llvm_target(AGXCodeGen
  AGXAnnotateControlFlow.cpp
  AGXAsmPrinter.cpp
  AGXCodeGenPrepare.cpp
  AGXFoldOperands.cpp
  AGXFrameLowering.cpp
  AGXISelDAGToDAG.cpp
  AGXISelLowering.cpp
  AGXInsertWaits.cpp
  AGXInstrInfo.cpp
  AGXLowerControlFlow.cpp
  AGXMachineFunctionInfo.cpp
  AGXMCInstLower.cpp
  AGXRegisterInfo.cpp
  AGXSelectionDAGInfo.cpp
  AGXSubtarget.cpp
  AGXTargetMachine.cpp
  AGXTargetTransformInfo.cpp

  LINK_COMPONENTS
  AGXDesc
  AGXInfo
  AGXUtils
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
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
  AGX
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
