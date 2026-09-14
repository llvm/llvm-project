# When building shared objects for each target there are some internal APIs
# that are used across shared objects which we can't hide.
if (NOT BUILD_SHARED_LIBS AND NOT APPLE AND
    (NOT (WIN32 OR CYGWIN) OR ((MINGW OR CYGWIN) AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")) AND
    NOT ("${CMAKE_SYSTEM_NAME}" MATCHES "AIX") AND
    NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
  # Set default visibility to hidden, so we don't export all the Target classes
  # in libLLVM.so.
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
endif()

add_llvm_component_library(LLVMGlobalISel
  CSEInfo.cpp
  GISelValueTracking.cpp
  CSEMIRBuilder.cpp
  CallLowering.cpp
  GlobalISel.cpp
  Combiner.cpp
  CombinerHelper.cpp
  CombinerHelperArtifacts.cpp
  CombinerHelperCasts.cpp
  CombinerHelperCompares.cpp
  CombinerHelperVectorOps.cpp
  GIMatchTableExecutor.cpp
  GISelChangeObserver.cpp
  IRTranslator.cpp
  InlineAsmLowering.cpp
  InstructionSelect.cpp
  InstructionSelector.cpp
  LegalityPredicates.cpp
  LegalizeMutations.cpp
  Legalizer.cpp
  LegalizerHelper.cpp
  LegalizerInfo.cpp
  LoadStoreOpt.cpp
  Localizer.cpp
  LostDebugLocObserver.cpp
  MachineFloatingPointPredicateUtils.cpp
  MachineIRBuilder.cpp
  RegBankSelect.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CodeGen/GlobalISel

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  )
