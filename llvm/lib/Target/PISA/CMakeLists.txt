add_llvm_component_group(PISA)

set(LLVM_TARGET_DEFINITIONS PISA.td)

tablegen(LLVM PISAGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM PISAGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM PISAGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM PISAGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(PISACommonTableGen)

add_llvm_target(PISACodeGen
  PISAInstrInfo.cpp
  PISARegisterInfo.cpp
  PISASubtarget.cpp
  PISATargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  PISADesc
  PISAInfo

  ADD_TO_COMPONENT
  PISA
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
