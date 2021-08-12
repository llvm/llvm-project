set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

add_tablegen(mlir-tblgen MLIR
  AttrOrTypeDefGen.cpp
  CodeGenHelpers.cpp
  DialectGen.cpp
  DirectiveCommonGen.cpp
  EnumsGen.cpp
  LLVMIRConversionGen.cpp
  LLVMIRIntrinsicGen.cpp
  mlir-tblgen.cpp
  OpDefinitionsGen.cpp
  OpDocGen.cpp
  OpFormatGen.cpp
  OpGenHelpers.cpp
  OpInterfacesGen.cpp
  OpPythonBindingGen.cpp
  PassCAPIGen.cpp
  PassDocGen.cpp
  PassGen.cpp
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
