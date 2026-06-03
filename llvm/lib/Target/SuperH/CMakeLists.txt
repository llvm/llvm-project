add_llvm_component_group(SuperH)

set(LLVM_TARGET_DEFINITIONS SuperH.td)

tablegen(LLVM SuperHGenAsmMatcher.inc         -gen-asm-matcher)
tablegen(LLVM SuperHGenAsmWriter.inc          -gen-asm-writer)
tablegen(LLVM SuperHGenCallingConv.inc        -gen-callingconv)
tablegen(LLVM SuperHGenDAGISel.inc            -gen-dag-isel)
tablegen(LLVM SuperHGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM SuperHGenRegisterInfo.inc       -gen-register-info)
tablegen(LLVM SuperHGenInstrInfo.inc          -gen-instr-info)
#tablegen(LLVM SuperHGenMCCodeEmitter.inc      -gen-emitter)
tablegen(LLVM SuperHGenSDNodeInfo.inc         -gen-sd-node-info -sdnode-namespace=SHISD)
tablegen(LLVM SuperHGenSearchableTables.inc   -gen-searchable-tables)
tablegen(LLVM SuperHGenSubtargetInfo.inc      -gen-subtarget)

add_public_tablegen_target(SuperHCommonTableGen)

add_llvm_target(SuperHCodeGen
  SuperHTargetMachine.cpp
  SuperHFrameLowering.cpp
  SuperHRegisterInfo.cpp
  SuperHInstrInfo.cpp
  SuperHSubtarget.cpp

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