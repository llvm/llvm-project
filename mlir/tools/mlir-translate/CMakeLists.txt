get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS
  ${dialect_libs}
  ${translation_libs}
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
)
add_llvm_tool(mlir-translate
  mlir-translate.cpp
)
llvm_update_compile_flags(mlir-translate)
target_link_libraries(mlir-translate PRIVATE MLIRIR MLIRTranslation ${LIBS} LLVMSupport)
