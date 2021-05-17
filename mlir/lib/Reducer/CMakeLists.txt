add_mlir_library(MLIRReduce
   Tester.cpp
   LINK_LIBS PUBLIC
   MLIRIR
)

mlir_check_all_link_libraries(MLIRReduce)
