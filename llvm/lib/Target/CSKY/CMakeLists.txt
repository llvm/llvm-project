add_llvm_component_group(CSKY)

set(LLVM_TARGET_DEFINITIONS CSKY.td)

tablegen(LLVM CSKYGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM CSKYGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM CSKYGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM CSKYGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM CSKYGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM CSKYGenSubtargetInfo.inc -gen-subtarget)

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

add_subdirectory(AsmParser)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
