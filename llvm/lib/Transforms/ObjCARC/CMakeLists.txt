add_llvm_component_library(LLVMObjCARCOpts
  ObjCARC.cpp
  ObjCARCOpts.cpp
  ObjCARCExpand.cpp
  ObjCARCContract.cpp
  DependencyAnalysis.cpp
  ProvenanceAnalysis.cpp
  ProvenanceAnalysisEvaluator.cpp
  PtrState.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms

  DEPENDS
  intrinsics_gen

  COMPONENT_NAME
  ObjCARC

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TargetParser
  TransformUtils
  )

# Static optimization functions conflict with other ObjCARC files.
set_source_files_properties(
  ObjCARCContract.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
