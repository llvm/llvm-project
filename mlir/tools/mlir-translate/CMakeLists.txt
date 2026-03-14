set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_mlir_tool(mlir-translate
  mlir-translate.cpp
  )
llvm_update_compile_flags(mlir-translate)
mlir_target_link_libraries(mlir-translate
  PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  )
target_link_libraries(mlir-translate
  PRIVATE
  ${translation_libs}
  ${test_libs}
  )

mlir_check_link_libraries(mlir-translate)
