set(LLVM_TARGET_DEFINITIONS Mips.td)

llvm_tablegen(MipsGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(MipsGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(MipsGenCodeEmitter.inc -gen-emitter)
llvm_tablegen(MipsGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(MipsGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(MipsGenCallingConv.inc -gen-callingconv)
llvm_tablegen(MipsGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(MipsCommonTableGen)

add_llvm_target(MipsCodeGen
  MipsAsmPrinter.cpp
  MipsCodeEmitter.cpp
  MipsDelaySlotFiller.cpp
  MipsEmitGPRestore.cpp
  MipsExpandPseudo.cpp
  MipsJITInfo.cpp
  MipsInstrInfo.cpp
  MipsISelDAGToDAG.cpp
  MipsISelLowering.cpp
  MipsFrameLowering.cpp
  MipsMCInstLower.cpp
  MipsMCSymbolRefExpr.cpp
  MipsRegisterInfo.cpp
  MipsSubtarget.cpp
  MipsTargetMachine.cpp
  MipsTargetObjectFile.cpp
  MipsSelectionDAGInfo.cpp
  )

add_llvm_library_dependencies(LLVMMipsCodeGen
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMMipsAsmPrinter
  LLVMMipsDesc
  LLVMMipsInfo
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
