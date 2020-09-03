get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AsmParser
  Core
  IRReader
  Support
  Target
  TransformUtils
  )

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${test_libs}
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIRIR
  MLIRLoopAnalysis
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRReduce
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  )

add_llvm_tool(mlir-reduce
  mlir-reduce.cpp
  )

target_link_libraries(mlir-reduce PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-reduce)

mlir_check_all_link_libraries(mlir-reduce)