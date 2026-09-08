add_mlir_library(MLIRReduce
   OptReductionPass.cpp
   ReductionNode.cpp
   ReductionTreePass.cpp
   Tester.cpp

   DEPENDS
   MLIRReducerIncGen
   MLIRDialectReductionPatternInterfaceIncGen

   LINK_LIBS PUBLIC
   MLIRIR
   MLIRPass
   MLIRRewrite
   MLIRTransformUtils

)

mlir_check_all_link_libraries(MLIRReduce)
