add_llvm_executable(mlir-edsc-builder-api-test
  builder-api-test.cpp
)

llvm_update_compile_flags(mlir-edsc-builder-api-test)

target_link_libraries(mlir-edsc-builder-api-test
  PRIVATE
  MLIRAffine
  MLIREDSC
  MLIRIR
  MLIRLinalgEDSC
  MLIRLinalgOps
  MLIRLoopOps
  MLIRStandardOps
  MLIRTransforms
  MLIRVector
  LLVMCore
  LLVMSupport
)

target_include_directories(mlir-edsc-builder-api-test PRIVATE ..)

whole_archive_link(mlir-edsc-builder-api-test
  MLIRAffine
  MLIRLinalgOps
  MLIRLoopOps
  MLIRStandardOps
  MLIRVector
  MLIRTransforms
)
