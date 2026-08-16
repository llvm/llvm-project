set(LIBS
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRLinalgDialect
    MLIRLinalgTransformOps
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTransformDialect
    MLIRTransformDialectTransforms
    MLIROptLib
    MLIRTileReducer
  )

add_llvm_executable(tr-opt tr-opt.cpp)
llvm_update_compile_flags(tr-opt)
target_link_libraries(tr-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tr-opt)
