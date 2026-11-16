add_llvm_component_group(CSKY)

set(LLVM_TARGET_DEFINITIONS CSKY.td)

llvm_tablegen(CSKYGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(CSKYGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(CSKYGenCallingConv.inc -gen-callingconv)
llvm_tablegen(CSKYGenCompressInstEmitter.inc -gen-compress-inst-emitter)
llvm_tablegen(CSKYGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(CSKYGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(CSKYGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(CSKYGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(CSKYGenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(CSKYGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(CSKYGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(CSKYGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS MCTargetDesc/CSKYMCOptions.td)
llvm_tablegen(CSKYMCOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(CSKYCommonTableGen)

add_llvm_target(CSKYCodeGen
  CSKYAsmPrinter.cpp
  CSKYConstantIslandPass.cpp
  CSKYConstantPoolValue.cpp
  CSKYFrameLowering.cpp
  CSKYInstrInfo.cpp
  CSKYISelDAGToDAG.cpp
  CSKYISelLowering.cpp
  CSKYMCInstLower.cpp
  CSKYRegisterInfo.cpp
  CSKYSelectionDAGInfo.cpp
  CSKYSubtarget.cpp
  CSKYTargetMachine.cpp
  CSKYTargetObjectFile.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CSKYDesc
  CSKYInfo
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser

  ADD_TO_COMPONENT
  CSKY
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
