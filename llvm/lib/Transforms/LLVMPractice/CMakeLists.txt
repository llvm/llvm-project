add_llvm_library( LLVMPractice MODULE BUILDTREE_ONLY
   myPeepHole.cpp

  PLUGIN_TOOL
  opt
  )
