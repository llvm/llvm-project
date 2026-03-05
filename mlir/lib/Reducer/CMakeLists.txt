add_mlir_library(MLIRReduce
   OptReductionPass.cpp
   ReductionNode.cpp
   ReductionTreePass.cpp
   Tester.cpp
   ReductionPatterns.cpp

   LINK_LIBS PUBLIC
   MLIRIR
   MLIRPass
   MLIRRewrite
   MLIRTransformUtils
   MLIRVectorDialect
   MLIRUBDialect

   DEPENDS
   MLIRReducerIncGen
   MLIRDialectReductionPatternInterfaceIncGen
)

mlir_check_all_link_libraries(MLIRReduce)