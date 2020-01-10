set(LLVM_TARGET_DEFINITIONS VE.td)

add_llvm_target(VECodeGen
  VETargetMachine.cpp
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
