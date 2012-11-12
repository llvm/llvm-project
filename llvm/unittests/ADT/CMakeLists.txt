set(LLVM_LINK_COMPONENTS
  Support
  )

set(ADTSources
  APFloatTest.cpp
  APIntTest.cpp
  BitVectorTest.cpp
  DAGDeltaAlgorithmTest.cpp
  DeltaAlgorithmTest.cpp
  DenseMapTest.cpp
  DenseSetTest.cpp
  FoldingSet.cpp
  HashingTest.cpp
  ilistTest.cpp
  ImmutableMapTest.cpp
  ImmutableSetTest.cpp
  IntEqClassesTest.cpp
  IntervalMapTest.cpp
  IntrusiveRefCntPtrTest.cpp
  PackedVectorTest.cpp
  SCCIteratorTest.cpp
  SmallPtrSetTest.cpp
  SmallStringTest.cpp
  SmallVectorTest.cpp
  SparseBitVectorTest.cpp
  SparseSetTest.cpp
  StringMapTest.cpp
  StringRefTest.cpp
  TinyPtrVectorTest.cpp
  TripleTest.cpp
  TwineTest.cpp
  VariadicFunctionTest.cpp
 )

# They cannot be compiled on MSVC9 due to its bug.
if(MSVC AND MSVC_VERSION LESS 1600)
  set(LLVM_OPTIONAL_SOURCES
    DenseMapTest.cpp
    SmallVectorTest.cpp
    )
  list(REMOVE_ITEM ADTSources ${LLVM_OPTIONAL_SOURCES})
endif()

add_llvm_unittest(ADTTests
  ${ADTSources}
  )
