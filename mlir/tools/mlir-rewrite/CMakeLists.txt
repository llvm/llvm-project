get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LLVM_LINK_COMPONENTS
  Support
  )

set(LIBS
  ${dialect_libs}
  ${test_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

include_directories(../../../clang/include)

add_mlir_tool(mlir-rewrite
  mlir-rewrite.cpp

  DEPENDS
  ${LIBS}
  SUPPORT_PLUGINS
  )
target_link_libraries(mlir-rewrite PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-rewrite)

mlir_check_all_link_libraries(mlir-rewrite)
export_executable_symbols_for_plugins(mlir-rewrite)
