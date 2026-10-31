add_llvm_component_group(ARC)

set(LLVM_TARGET_DEFINITIONS ARC.td)

llvm_tablegen(ARCGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(ARCGenCallingConv.inc -gen-callingconv)
llvm_tablegen(ARCGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(ARCGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(ARCGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(ARCGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(ARCGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(ARCGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(ARCCommonTableGen)

add_llvm_target(ARCCodeGen
  ARCAsmPrinter.cpp
  ARCBranchFinalize.cpp
  ARCExpandPseudos.cpp
  ARCFrameLowering.cpp
  ARCInstrInfo.cpp
  ARCISelDAGToDAG.cpp
  ARCISelLowering.cpp
  ARCMachineFunctionInfo.cpp
  ARCMCInstLower.cpp
  ARCOptAddrMode.cpp
  ARCRegisterInfo.cpp
  ARCSelectionDAGInfo.cpp
  ARCSubtarget.cpp
  ARCTargetMachine.cpp

  LINK_COMPONENTS
  ARCDesc
  ARCInfo
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

  ADD_TO_COMPONENT
  ARC
  )

add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
