
add_tablegen(mlir-tblgen MLIR
  DialectGen.cpp
  EnumsGen.cpp
  LLVMIRConversionGen.cpp
  LLVMIRIntrinsicGen.cpp
  mlir-tblgen.cpp
  OpDefinitionsGen.cpp
  OpDocGen.cpp
  OpFormatGen.cpp
  OpInterfacesGen.cpp
  OpenMPCommonGen.cpp
  PassGen.cpp
  PassDocGen.cpp
  RewriterGen.cpp
  SPIRVUtilsGen.cpp
  StructsGen.cpp
  )

set_target_properties(mlir-tblgen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(mlir-tblgen
  PRIVATE
  MLIRSupportIndentedOstream
  MLIRTableGen)

mlir_check_all_link_libraries(mlir-tblgen)
