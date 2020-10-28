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

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRAffineTransformsTestPasses
    MLIRSPIRVTestPasses
    MLIRTestDialect
    MLIRTestIR
    MLIRTestPass
    MLIRTestReducer
    MLIRTestTransforms
    )
endif()

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
  OptReductionPass.cpp
  Passes/OpReducer.cpp
  ReductionNode.cpp
  ReductionTreeUtils.cpp
  mlir-reduce.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Reducer

  DEPENDS
  MLIRReducerIncGen
  )

target_link_libraries(mlir-reduce PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-reduce)

mlir_check_all_link_libraries(mlir-reduce)
