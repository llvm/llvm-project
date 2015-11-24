add_llvm_library(LLVMipo
  ArgumentPromotion.cpp
  BarrierNoopPass.cpp
  ConstantMerge.cpp
  DeadArgumentElimination.cpp
  ElimAvailExtern.cpp
  ExtractGV.cpp
  FunctionAttrs.cpp
  FunctionImport.cpp
  GlobalDCE.cpp
  GlobalOpt.cpp
  IPConstantPropagation.cpp
  IPO.cpp
  InlineAlways.cpp
  InlineSimple.cpp
  Inliner.cpp
  Internalize.cpp
  LoopExtractor.cpp
  LowerBitSets.cpp
  MergeFunctions.cpp
  PartialInlining.cpp
  PassManagerBuilder.cpp
  PruneEH.cpp
  SampleProfile.cpp
  StripDeadPrototypes.cpp
  StripSymbols.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/IPO
  )

add_dependencies(LLVMipo intrinsics_gen)
