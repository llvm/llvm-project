set(LLVM_OPTIONAL_SOURCES
  ir.c
  pass.c
)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_executable(mlir-capi-ir-test
  ir.c
  )
llvm_update_compile_flags(mlir-capi-ir-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-capi-ir-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegistration
  MLIRCAPIStandard
  ${dialect_libs})


add_llvm_executable(mlir-capi-pass-test
  pass.c
  )
llvm_update_compile_flags(mlir-capi-pass-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-capi-pass-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegistration
  MLIRCAPIStandard
  MLIRCAPITransforms
  ${dialect_libs})
