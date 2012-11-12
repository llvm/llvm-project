set(LLVM_TARGET_DEFINITIONS Sparc.td)

tablegen(LLVM SparcGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SparcGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SparcGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SparcGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM SparcGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM SparcGenCallingConv.inc -gen-callingconv)
add_public_tablegen_target(SparcCommonTableGen)

add_llvm_target(SparcCodeGen
  DelaySlotFiller.cpp
  FPMover.cpp
  SparcAsmPrinter.cpp
  SparcInstrInfo.cpp
  SparcISelDAGToDAG.cpp
  SparcISelLowering.cpp
  SparcFrameLowering.cpp
  SparcMachineFunctionInfo.cpp
  SparcRegisterInfo.cpp
  SparcSubtarget.cpp
  SparcTargetMachine.cpp
  SparcSelectionDAGInfo.cpp
  )

add_dependencies(LLVMSparcCodeGen intrinsics_gen)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
