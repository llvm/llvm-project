add_llvm_component_library(LLVMSelectionDAG
  DAGCombiner.cpp
  FastISel.cpp
  FunctionLoweringInfo.cpp
  InstrEmitter.cpp
  LegalizeDAG.cpp
  LegalizeFloatTypes.cpp
  LegalizeIntegerTypes.cpp
  LegalizeTypes.cpp
  LegalizeTypesGeneric.cpp
  LegalizeVectorOps.cpp
  LegalizeVectorTypes.cpp
  ResourcePriorityQueue.cpp
  SDNodeInfo.cpp
  ScheduleDAGFast.cpp
  ScheduleDAGRRList.cpp
  ScheduleDAGSDNodes.cpp
  ScheduleDAGVLIW.cpp
  SelectionDAGBuilder.cpp
  SelectionDAG.cpp
  SelectionDAGAddressAnalysis.cpp
  SelectionDAGDumper.cpp
  SelectionDAGISel.cpp
  SelectionDAGPrinter.cpp
  SelectionDAGTargetInfo.cpp
  StatepointLowering.cpp
  TargetLowering.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  CodeGen
  CodeGenTypes
  Core
  MC
  Support
  Target
  TargetParser
  TransformUtils
  )

# STATISTIC variable redefinitions (NumBacktracks, NumUnfolds, etc.)
# and static function conflicts.
set_source_files_properties(
  ScheduleDAGRRList.cpp
  SelectionDAG.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# Large template instantiation code and static helper conflicts.
set_source_files_properties(
  TargetLowering.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
