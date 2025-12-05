set(LLVM_LINK_COMPONENTS
  Ripple
  ExecutionEngine
  AllTargetsCodeGens
  AllTargetsAsmParsers
  Passes
  )

add_llvm_unittest(RippleTests
  TensorShapeTests.cpp
  PropagateShapeTests.cpp
  )
