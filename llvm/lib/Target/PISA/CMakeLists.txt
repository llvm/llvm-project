add_llvm_component_group(PISA)

set(LLVM_TARGET_DEFINITIONS PISA.td)
tablegen(LLVM PISAGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM PISAGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM PISAGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM PISAGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM PISAGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM PISAGenSearchableTables.inc -gen-searchable-tables)

add_public_tablegen_target(PISACommonTableGen)

add_llvm_target(PISACodeGen
  PISAInstrInfo.cpp
  PISAMCInstLower.cpp
  PISARegManager.cpp
  PISARegisterInfo.cpp
  PISASubtarget.cpp
  PISATargetMachine.cpp
  PISAUtils.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  Demangle
  GlobalISel
  MC
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  PISADesc
  PISAInfo

  ADD_TO_COMPONENT
  PISA
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
