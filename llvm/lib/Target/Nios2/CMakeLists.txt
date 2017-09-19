set(LLVM_TARGET_DEFINITIONS Nios2.td)

#Generate Nios2GenRegisterInfo.inc and Nios2GenInstrInfo.inc which included by
#your hand code C++ files.
#Nios2GenRegisterInfo.inc came from Nios2RegisterInfo.td, Nios2GenInstrInfo.inc
#came from Nios2InstrInfo.td.
tablegen(LLVM Nios2GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM Nios2GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM Nios2GenSubtargetInfo.inc -gen-subtarget)

#Nios2CommonTableGen must be defined
add_public_tablegen_target(Nios2CommonTableGen)

#Nios2CodeGen should match with LLVMBuild.txt Nios2CodeGen
add_llvm_target(Nios2CodeGen
 Nios2InstrInfo.cpp
 Nios2FrameLowering.cpp
 Nios2RegisterInfo.cpp
 Nios2Subtarget.cpp
 Nios2TargetMachine.cpp
 )

#Should match with "subdirectories =  MCTargetDesc TargetInfo" in LLVMBuild.txt
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
