add_llvm_component_group(Sparc)

set(LLVM_TARGET_DEFINITIONS Sparc.td)

llvm_tablegen(SparcGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(SparcGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(SparcGenCallingConv.inc -gen-callingconv)
llvm_tablegen(SparcGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(SparcGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(SparcGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(SparcGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(SparcGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(SparcGenSDNodeInfo.inc -gen-sd-node-info -sdnode-namespace=SPISD)
llvm_tablegen(SparcGenSearchableTables.inc -gen-searchable-tables)
llvm_tablegen(SparcGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS SparcOptions.td)
llvm_tablegen(SparcOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(SparcCommonTableGen)

add_llvm_target(SparcCodeGen
  DelaySlotFiller.cpp
  LeonPasses.cpp
  SparcAsmPrinter.cpp
  SparcInstrInfo.cpp
  SparcISelDAGToDAG.cpp
  SparcISelLowering.cpp
  SparcFrameLowering.cpp
  SparcMachineFunctionInfo.cpp
  SparcRegisterInfo.cpp
  SparcSelectionDAGInfo.cpp
  SparcSubtarget.cpp
  SparcTargetMachine.cpp
  SparcTargetObjectFile.cpp
  SparcTargetTransformInfo.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  Option
  SelectionDAG
  SparcDesc
  SparcInfo
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Sparc
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
