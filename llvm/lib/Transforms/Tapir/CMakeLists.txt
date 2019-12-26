add_llvm_library(LLVMTapirOpts
  AnalyzeTapir.cpp
  CilkABI.cpp
  CilkRABI.cpp
  DRFScopedNoAliasAA.cpp
  LoopSpawningTI.cpp
  LoopStripMine.cpp
  LoopStripMinePass.cpp
  LoweringUtils.cpp
  OpenMPABI.cpp
  Outline.cpp
  RedundantSpawn.cpp
  SerializeSmallTasks.cpp
  SmallBlock.cpp
  SpawnRestructure.cpp
  SpawnUnswitch.cpp
  SyncElimination.cpp
  Tapir.cpp
  TapirToTarget.cpp
  TapirLoopInfo.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir
  )

add_dependencies(LLVMTapirOpts intrinsics_gen)
