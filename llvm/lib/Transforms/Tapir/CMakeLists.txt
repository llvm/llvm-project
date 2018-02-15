add_llvm_library(LLVMTapirOpts
  CilkABI.cpp
  CilkRABI.cpp
  LoopSpawning.cpp
  LoweringUtils.cpp
  OpenMPABI.cpp
  Outline.cpp
  RedundantSpawn.cpp
  SmallBlock.cpp
  SpawnRestructure.cpp
  SpawnUnswitch.cpp
  SyncElimination.cpp
  Tapir.cpp
  TapirToTarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir
  )

add_dependencies(LLVMTapirOpts intrinsics_gen)
