add_mlir_library(MLIRReduce
   OptReductionPass.cpp
   ReductionNode.cpp
   ReductionTreePass.cpp
   Tester.cpp

   LINK_LIBS PUBLIC
   MLIRIR
   MLIRPass
   MLIRRewrite
   MLIRTransformUtils

   DEPENDS
   MLIRReducerIncGen
)

mlir_check_all_link_libraries(MLIRReduce)
