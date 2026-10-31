add_llvm_component_group(VE)

set(LLVM_TARGET_DEFINITIONS VE.td)

llvm_tablegen(VEGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(VEGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(VEGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(VEGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(VEGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(VEGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(VEGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(VEGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(VEGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(VEGenCallingConv.inc -gen-callingconv)
add_public_tablegen_target(VECommonTableGen)

add_llvm_target(VECodeGen
  LVLGen.cpp
  VEAsmPrinter.cpp
  VECustomDAG.cpp
  VEFrameLowering.cpp
  VEISelDAGToDAG.cpp
  VEISelLowering.cpp
  VEInstrInfo.cpp
  VEMachineFunctionInfo.cpp
  VEMCInstLower.cpp
  VERegisterInfo.cpp
  VESelectionDAGInfo.cpp
  VESubtarget.cpp
  VETargetMachine.cpp
  VVPISelLowering.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  VEDesc
  VEInfo

  ADD_TO_COMPONENT
  VE
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
