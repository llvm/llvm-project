set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  BitReader
  BitWriter
  CodeGen
  Core
  Coroutines
  IPO
  IRReader
  InstCombine
  Instrumentation
  FuzzMutate
  MC
  ObjCARCOpts
  ScalarOpts
  Support
  Target
  TransformUtils
  Vectorize
  Passes
)

add_llvm_fuzzer(llvm-opt-fuzzer llvm-opt-fuzzer.cpp
  DUMMY_MAIN DummyOptFuzzer.cpp)
