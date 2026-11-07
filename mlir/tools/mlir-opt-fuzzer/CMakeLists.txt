set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
)
add_llvm_fuzzer(mlir-opt-fuzzer
  mlir-opt-fuzzer.cpp
  DUMMY_MAIN DummyOptFuzzer.cpp
)
mlir_target_link_libraries(mlir-opt-fuzzer
  PUBLIC
  MLIRFuzzMutate
  MLIRIR
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSupport
)
