add_mlir_unittest(MLIRToolsMlirOptContextPreload
  MlirOptContextPreload.cpp
)

target_link_libraries(MLIRToolsMlirOptContextPreload
  PRIVATE
  MLIROptLib
  MLIRParser
  MLIRTransformDialect
  MLIRTestTransformDialect
)
