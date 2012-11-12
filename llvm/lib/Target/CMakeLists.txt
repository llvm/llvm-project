add_llvm_library(LLVMTarget
  Mangler.cpp
  Target.cpp
  TargetInstrInfo.cpp
  TargetIntrinsicInfo.cpp
  TargetJITInfo.cpp
  TargetLibraryInfo.cpp
  TargetLoweringObjectFile.cpp
  TargetMachine.cpp
  TargetMachineC.cpp
  TargetRegisterInfo.cpp
  TargetSubtargetInfo.cpp
  TargetTransformImpl.cpp
  )

foreach(t ${LLVM_TARGETS_TO_BUILD})
  message(STATUS "Targeting ${t}")
  add_subdirectory(${t})
endforeach()
