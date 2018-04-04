set(LLVM_TARGET_DEFINITIONS Nios2.td)

tablegen(LLVM Nios2GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM Nios2GenCallingConv.inc -gen-callingconv)
tablegen(LLVM Nios2GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM Nios2GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM Nios2GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM Nios2GenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(Nios2CommonTableGen)

#Nios2CodeGen should match with LLVMBuild.txt Nios2CodeGen
add_llvm_target(Nios2CodeGen
 Nios2AsmPrinter.cpp
 Nios2FrameLowering.cpp
 Nios2InstrInfo.cpp
 Nios2ISelDAGToDAG.cpp
 Nios2ISelLowering.cpp
 Nios2MachineFunction.cpp
 Nios2MCInstLower.cpp
 Nios2RegisterInfo.cpp
 Nios2Subtarget.cpp
 Nios2TargetMachine.cpp
 Nios2TargetObjectFile.cpp
 )

#Should match with "subdirectories =  InstPrinter MCTargetDesc TargetInfo" in LLVMBuild.txt
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
