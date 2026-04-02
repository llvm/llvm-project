set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(mlir-tblgen MLIR
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  AttrOrTypeDefGen.cpp
  AttrOrTypeFormatGen.cpp
  BytecodeDialectGen.cpp
  DialectGen.cpp
  DialectInterfacesGen.cpp
  DirectiveCommonGen.cpp
  EnumsGen.cpp
  EnumPythonBindingGen.cpp
  FormatGen.cpp
  LLVMIRConversionGen.cpp
  LLVMIRIntrinsicGen.cpp
  mlir-tblgen.cpp
  OmpOpGen.cpp
  OpClass.cpp
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
  TosaUtilsGen.cpp
  CppGenUtilities.cpp
  )

# Unity build conflicts:
# - BytecodeDialectGen.cpp/DialectGen.cpp: both define static 'dialectGenCat'
# - PassGen.cpp/PassCAPIGen.cpp: both define 'passGenCat', 'groupName',
#   'passGroupRegistrationCode', 'fileHeader'
# - SPIRVUtilsGen.cpp/TosaUtilsGen.cpp: both define 'Availability' class and
#   related functions in anonymous namespaces
# - LLVMIRIntrinsicGen.cpp, OpDocGen.cpp, OpPythonBindingGen.cpp, RewriterGen.cpp:
#   'using namespace llvm;' from other files makes DenseMap, Type, StringSwitch,
#   SetVector ambiguous with mlir::tblgen equivalents
set_source_files_properties(
  BytecodeDialectGen.cpp
  DialectGen.cpp
  LLVMIRIntrinsicGen.cpp
  OpDocGen.cpp
  OpFormatGen.cpp
  OpPythonBindingGen.cpp
  PassCAPIGen.cpp
  PassGen.cpp
  RewriterGen.cpp
  SPIRVUtilsGen.cpp
  TosaUtilsGen.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

target_link_libraries(mlir-tblgen
  PRIVATE
  MLIRTblgenLib)

mlir_check_all_link_libraries(mlir-tblgen)
