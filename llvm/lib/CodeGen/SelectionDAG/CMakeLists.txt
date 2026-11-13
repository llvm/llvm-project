set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)
set(LLVM_TARGET_DEFINITIONS SelectionDAGOptions.td)
llvm_tablegen(SelectionDAGOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(SelectionDAGOptionsTableGen)

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
  SelectionDAGOptions.cpp
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
  Option
  Support
  Target
  TargetParser
  TransformUtils
  )
