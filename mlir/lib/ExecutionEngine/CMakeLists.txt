set(LLVM_OPTIONAL_SOURCES
  ExecutionEngine.cpp
  RunnerUtils.cpp
  OptUtils.cpp
  )

llvm_map_components_to_libnames(outlibs "nativecodegen" "IPO")
add_llvm_library(MLIRExecutionEngine
  ExecutionEngine.cpp
  OptUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ExecutionEngine
  )
target_link_libraries(MLIRExecutionEngine

  MLIRLLVMIR
  MLIRTargetLLVMIR
  LLVMBitReader
  LLVMBitWriter
  LLVMExecutionEngine
  LLVMObject
  LLVMOrcJIT
  LLVMJITLink
  LLVMSupport
  LLVMAnalysis
  LLVMAggressiveInstCombine
  LLVMInstCombine
  LLVMMC
  LLVMScalarOpts
  LLVMTarget
  LLVMVectorize
  LLVMTransformUtils

  ${outlibs})

add_llvm_library(mlir_runner_utils SHARED RunnerUtils.cpp)
target_compile_definitions(mlir_runner_utils PRIVATE mlir_runner_utils_EXPORTS)
