add_llvm_library(LLVMTarget
  DarwinTargetAsmInfo.cpp
  ELFTargetAsmInfo.cpp
  SubtargetFeature.cpp
  Target.cpp
  TargetAsmInfo.cpp
  TargetData.cpp
  TargetFrameInfo.cpp
  TargetInstrInfo.cpp
  TargetMachOWriterInfo.cpp
  TargetMachine.cpp
  TargetMachineRegistry.cpp
  TargetRegisterInfo.cpp
  TargetSubtarget.cpp
  )

# TODO: Support other targets besides X86. See Makefile.