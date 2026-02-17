add_dependencies(DataFlowExample dataflow-opt)
add_llvm_example(dataflow-opt
  dataflow-opt.cpp
)

target_link_libraries(dataflow-opt
  PRIVATE
  MLIRIR
  MLIRMlirOptMain
  MLIRStringDialect
  MLIRTestStringConstantPropagation
  MLIRTestMetadataAnalysisPass
)
