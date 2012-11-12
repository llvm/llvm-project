set(LLVM_TARGET_DEFINITIONS SPU.td)

tablegen(LLVM SPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SPUGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM SPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM SPUGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM SPUGenCallingConv.inc -gen-callingconv)
add_public_tablegen_target(CellSPUCommonTableGen)

add_llvm_target(CellSPUCodeGen
  SPUAsmPrinter.cpp
  SPUHazardRecognizers.cpp
  SPUInstrInfo.cpp
  SPUISelDAGToDAG.cpp
  SPUISelLowering.cpp
  SPUFrameLowering.cpp
  SPUMachineFunction.cpp
  SPURegisterInfo.cpp
  SPUSubtarget.cpp
  SPUTargetMachine.cpp
  SPUSelectionDAGInfo.cpp
  SPUNopFiller.cpp
  )

add_dependencies(LLVMCellSPUCodeGen intrinsics_gen)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
