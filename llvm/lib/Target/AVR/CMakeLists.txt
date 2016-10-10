set(LLVM_TARGET_DEFINITIONS AVR.td)

tablegen(LLVM AVRGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AVRGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AVRGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AVRGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AVRGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AVRGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(AVRCommonTableGen)

add_llvm_target(AVRCodeGen
  AVRAsmPrinter.cpp
  AVRFrameLowering.cpp
  AVRInstrInfo.cpp
  AVRMCInstLower.cpp
  AVRRegisterInfo.cpp
  AVRSubtarget.cpp
  AVRTargetMachine.cpp
  AVRTargetObjectFile.cpp
)

add_dependencies(LLVMAVRCodeGen intrinsics_gen)

add_subdirectory(AsmParser)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)

