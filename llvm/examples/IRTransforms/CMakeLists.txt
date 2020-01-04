set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  )

add_llvm_example_library(ExampleIRTransforms
  InitializePasses.cpp
  SimplifyCFG.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  intrinsics_gen
  )
