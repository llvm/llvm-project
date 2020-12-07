add_llvm_component_group(CSKY)

set(LLVM_TARGET_DEFINITIONS CSKY.td)

tablegen(LLVM CSKYGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM CSKYGenInstrInfo.inc -gen-instr-info)

add_public_tablegen_target(CSKYCommonTableGen)

add_llvm_target(CSKYCodeGen
  CSKYTargetMachine.cpp

  LINK_COMPONENTS
  Core
  CodeGen
  CSKYInfo
  Support
  Target

  ADD_TO_COMPONENT
  CSKY
  )

add_subdirectory(TargetInfo)
