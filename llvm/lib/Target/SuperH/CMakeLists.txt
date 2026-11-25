add_llvm_component_group(SuperH)

set(LLVM_TARGET_DEFINITIONS SuperH.td)

llvm_tablegen(SuperHGenAsmMatcher.inc         -gen-asm-matcher)
llvm_tablegen(SuperHGenAsmWriter.inc          -gen-asm-writer)
llvm_tablegen(SuperHGenDAGISel.inc            -gen-dag-isel)
llvm_tablegen(SuperHGenCallingConv.inc        -gen-callingconv)
llvm_tablegen(SuperHGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(SuperHGenRegisterInfo.inc       -gen-register-info)
llvm_tablegen(SuperHGenInstrInfo.inc          -gen-instr-info)
llvm_tablegen(SuperHGenMCCodeEmitter.inc      -gen-emitter)
llvm_tablegen(SuperHGenSDNodeInfo.inc         -gen-sd-node-info -sdnode-namespace=SHISD)
llvm_tablegen(SuperHGenSearchableTables.inc   -gen-searchable-tables)
llvm_tablegen(SuperHGenSubtargetInfo.inc      -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS SuperHOptions.td)
llvm_tablegen(SuperHOptions.inc -gen-opt-parser-defs)

set(LLVM_TARGET_DEFINITIONS MCTargetDesc/SuperHMCOptions.td)
llvm_tablegen(SuperHMCOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(SuperHCommonTableGen)

add_llvm_target(SuperHCodeGen
  SuperHTargetMachine.cpp
  SuperHFrameLowering.cpp
  SuperHConstantPoolValue.cpp
  SuperHBasicBlockInfo.cpp
  SuperHMachineFunctionInfo.cpp
  SuperHRegisterInfo.cpp
  SuperHInstrInfo.cpp
  SuperHSelectionDAGInfo.cpp
  SuperHISelLowering.cpp
  SuperHISelDAGToDAG.cpp
  SuperHMCInstLower.cpp
  SuperHAsmPrinter.cpp
  SuperHSubtarget.cpp
  SuperHFillDelaySlots.cpp
  SuperHConstantIslandPass.cpp
  SuperHExpandPseudoInsts.cpp
  SuperHExpandBranches.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  SuperHDesc
  SuperHInfo
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  SuperH
  )

add_subdirectory(AsmParser)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)