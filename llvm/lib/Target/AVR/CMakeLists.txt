set(LLVM_TARGET_DEFINITIONS AVR.td)

tablegen(LLVM AVRGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AVRGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AVRGenCallingConv.inc -gen-callingconv)
add_public_tablegen_target(AVRCommonTableGen)

add_llvm_target(AVRCodeGen
    AVRTargetMachine.cpp
    AVRTargetObjectFile.cpp
  )

add_dependencies(LLVMAVRCodeGen intrinsics_gen)

add_subdirectory(TargetInfo)

