set(LLVM_LINK_COMPONENTS
  MLIRTableGen
  Support
  )

add_tablegen(mlir-tblgen MLIR
  EnumsGen.cpp
  LLVMIRConversionGen.cpp
  LLVMIRIntrinsicGen.cpp
  mlir-tblgen.cpp
  OpDefinitionsGen.cpp
  OpDocGen.cpp
  OpFormatGen.cpp
  OpInterfacesGen.cpp
  ReferenceImplGen.cpp
  RewriterGen.cpp
  SPIRVUtilsGen.cpp
  StructsGen.cpp
  )
set_target_properties(mlir-tblgen PROPERTIES FOLDER "Tablegenning")
