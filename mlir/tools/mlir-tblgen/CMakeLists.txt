set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(mlir-tblgen MLIR
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  DirectiveCommonGen.cpp
  Generators.cpp
  LLVMIRConversionGen.cpp
  LLVMIRIntrinsicGen.cpp
  mlir-tblgen.cpp
  OmpOpGen.cpp
  SPIRVUtilsGen.cpp
  TosaUtilsGen.cpp
  )

target_link_libraries(mlir-tblgen
  PRIVATE
  MLIRTableGenGenerators
  MLIRTblgenLib)

mlir_check_all_link_libraries(mlir-tblgen)
