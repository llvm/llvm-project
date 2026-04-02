set(LLVM_LINK_COMPONENTS
  nativecodegen
  native
  orcjit
  support
)

add_mlir_unittest(MLIRExecutionEngineTests
  DynamicMemRef.cpp
  StridedMemRef.cpp
  Invoke.cpp
)

mlir_target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRRegisterAllDialects
)
target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRExecutionEngine
)

# These files include CRunnerUtils.h/RunnerUtils.h which defines
# UnrankedMemRefType that conflicts with mlir::UnrankedMemRefType in unity builds.
set_source_files_properties(
  DynamicMemRef.cpp
  StridedMemRef.cpp
  Invoke.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
