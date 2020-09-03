add_mlir_library(MLIRReduce
   Tester.cpp
   DEPENDS
   MLIRIR
 )
 
 mlir_check_all_link_libraries(MLIRReduce)