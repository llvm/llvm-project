set(LLVM_TARGET_DEFINITIONS Xtensa.td)

tablegen(LLVM XtensaGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM XtensaGenRegisterInfo.inc -gen-register-info)

add_public_tablegen_target(XtensaCommonTableGen) 

add_llvm_target(XtensaCodeGen
  XtensaTargetMachine.cpp
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)

