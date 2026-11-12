add_llvm_component_group(Lanai)

set(LLVM_TARGET_DEFINITIONS Lanai.td)

llvm_tablegen(LanaiGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(LanaiGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(LanaiGenCallingConv.inc -gen-callingconv)
llvm_tablegen(LanaiGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(LanaiGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(LanaiGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(LanaiGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(LanaiGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(LanaiGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(LanaiGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS LanaiOptions.td)
llvm_tablegen(LanaiOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(LanaiCommonTableGen)

add_llvm_target(LanaiCodeGen
  LanaiAsmPrinter.cpp
  LanaiCodeGenPassBuilder.cpp
  LanaiDelaySlotFiller.cpp
  LanaiFrameLowering.cpp
  LanaiInstrInfo.cpp
  LanaiISelDAGToDAG.cpp
  LanaiISelLowering.cpp
  LanaiMachineFunctionInfo.cpp
  LanaiMCInstLower.cpp
  LanaiMemAluCombiner.cpp
  LanaiRegisterInfo.cpp
  LanaiSelectionDAGInfo.cpp
  LanaiSubtarget.cpp
  LanaiTargetMachine.cpp
  LanaiTargetObjectFile.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  LanaiAsmParser
  LanaiDesc
  LanaiInfo
  MC
  Option
  Passes
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Lanai
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
