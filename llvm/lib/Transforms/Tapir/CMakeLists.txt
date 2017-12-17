add_llvm_library(LLVMTapirOpts
  CilkABI.cpp
  CilkRABI.cpp
  OpenMPABI.cpp
  SmallBlock.cpp
  RedundantSpawn.cpp
  SpawnRestructure.cpp
  SpawnUnswitch.cpp
  SyncElimination.cpp
  TapirToTarget.cpp
  LoopSpawning.cpp
  Outline.cpp
  Tapir.cpp
  TapirUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir
  )

add_dependencies(LLVMTapirOpts intrinsics_gen)
