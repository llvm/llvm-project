set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

add_tablegen(mlir-pdll MLIR_PDLL
  mlir-pdll.cpp
  )

set_target_properties(mlir-pdll PROPERTIES FOLDER "Tablegenning")
target_link_libraries(mlir-pdll
  PRIVATE
  MLIRPDLLAST
  MLIRPDLLCodeGen
  MLIRPDLLParser
  )

mlir_check_all_link_libraries(mlir-pdll)
