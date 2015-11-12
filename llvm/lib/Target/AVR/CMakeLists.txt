

add_llvm_target(AVRCodeGen
    AVRTargetMachine.cpp
  )

add_dependencies(LLVMAVRCodeGen intrinsics_gen)

add_subdirectory(TargetInfo)

