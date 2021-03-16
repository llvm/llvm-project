set(LLVM_LINK_COMPONENTS
  Core
  Support
  )
add_llvm_executable(mlir-edsc-builder-api-test
  builder-api-test.cpp
)

llvm_update_compile_flags(mlir-edsc-builder-api-test)

target_link_libraries(mlir-edsc-builder-api-test
  PRIVATE
  MLIRAffine
  MLIRAffineEDSC
  MLIREDSC
  MLIRIR
  MLIRLinalg
  MLIRLinalgEDSC
  MLIRMemRef
  MLIRSCF
  MLIRStandard
  MLIRTransforms
  MLIRVector
  )

target_include_directories(mlir-edsc-builder-api-test PRIVATE ..)
