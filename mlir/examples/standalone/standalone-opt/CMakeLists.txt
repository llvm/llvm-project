get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithmetic
        MLIROptLib
        MLIRStandalone
        )
add_llvm_executable(standalone-opt standalone-opt.cpp)

llvm_update_compile_flags(standalone-opt)
target_link_libraries(standalone-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(standalone-opt)
