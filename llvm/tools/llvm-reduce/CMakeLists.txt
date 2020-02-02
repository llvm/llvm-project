set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Core
  IRReader
  Support
  Target
  TransformUtils
  )

add_llvm_tool(llvm-reduce
  llvm-reduce.cpp
  TestRunner.cpp
  deltas/Delta.cpp
  deltas/ReduceArguments.cpp
  deltas/ReduceAttributes.cpp
  deltas/ReduceBasicBlocks.cpp
  deltas/ReduceFunctions.cpp
  deltas/ReduceGlobalVars.cpp
  deltas/ReduceInstructions.cpp
  deltas/ReduceMetadata.cpp

  DEPENDS
  intrinsics_gen
  )
