add_llvm_component_group(XCore)

set(LLVM_TARGET_DEFINITIONS XCore.td)

llvm_tablegen(XCoreGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(XCoreGenCallingConv.inc -gen-callingconv)
llvm_tablegen(XCoreGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(XCoreGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(XCoreGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(XCoreGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(XCoreGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(XCoreGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS XCoreOptions.td)
llvm_tablegen(XCoreOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(XCoreCommonTableGen)

add_llvm_target(XCoreCodeGen
  XCoreAsmPrinter.cpp
  XCoreCodeGenPassBuilder.cpp
  XCoreFrameLowering.cpp
  XCoreInstrInfo.cpp
  XCoreISelDAGToDAG.cpp
  XCoreISelLowering.cpp
  XCoreLowerThreadLocal.cpp
  XCoreMachineFunctionInfo.cpp
  XCoreMCInstLower.cpp
  XCoreRegisterInfo.cpp
  XCoreSubtarget.cpp
  XCoreTargetMachine.cpp
  XCoreTargetObjectFile.cpp
  XCoreSelectionDAGInfo.cpp
  XCoreFrameToArgsOffsetElim.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  Option
  Passes
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  XCoreDesc
  XCoreInfo

  ADD_TO_COMPONENT
  XCore
  )

add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
