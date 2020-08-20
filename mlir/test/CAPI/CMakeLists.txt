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
  ${dialect_libs})
