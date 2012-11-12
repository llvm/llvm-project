add_llvm_library(LLVMipo
  ArgumentPromotion.cpp
  BarrierNoopPass.cpp
  ConstantMerge.cpp
  DeadArgumentElimination.cpp
  ExtractGV.cpp
  FunctionAttrs.cpp
  GlobalDCE.cpp
  GlobalOpt.cpp
  IPConstantPropagation.cpp
  IPO.cpp
  InlineAlways.cpp
  InlineSimple.cpp
  Inliner.cpp
  Internalize.cpp
  LoopExtractor.cpp
  MergeFunctions.cpp
  PartialInlining.cpp
  PassManagerBuilder.cpp
  PruneEH.cpp
  StripDeadPrototypes.cpp
  StripSymbols.cpp
  )

add_dependencies(LLVMipo intrinsics_gen)
