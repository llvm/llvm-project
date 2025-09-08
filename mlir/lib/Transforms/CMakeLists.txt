add_subdirectory(Utils)

add_mlir_library(MLIRTransforms
  Canonicalizer.cpp
  CompositePass.cpp
  ControlFlowSink.cpp
  CSE.cpp
  GenerateRuntimeVerification.cpp
  InlinerPass.cpp
  LocationSnapshot.cpp
  LoopInvariantCodeMotion.cpp
  Mem2Reg.cpp
  OpStats.cpp
  PrintIR.cpp
  RemoveDeadValues.cpp
  SCCP.cpp
  SROA.cpp
  StripDebugInfo.cpp
  SymbolDCE.cpp
  SymbolPrivatize.cpp
  TopologicalSort.cpp
  ViewOpGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRLoopLikeInterface
  MLIRMemorySlotInterfaces
  MLIRPass
  MLIRRuntimeVerifiableOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  )
