set(LLVM_LINK_COMPONENTS
    ${LLVM_TARGETS_TO_BUILD}
    Analysis
    AsmPrinter
    BitReader
    BitWriter
    CodeGen
    Core
    FuzzMutate
    IRReader
    MC
    ScalarOpts
    SelectionDAG
    Support
    Target
)
add_llvm_fuzzer(llvm-isel-fuzzer llvm-isel-fuzzer.cpp
  DUMMY_MAIN DummyISelFuzzer.cpp)
