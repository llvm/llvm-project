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
  deltas/ReduceFunctions.cpp
  deltas/ReduceGlobalVars.cpp
  deltas/ReduceMetadata.cpp
  deltas/ReduceArguments.cpp

  DEPENDS
  intrinsics_gen
  )
