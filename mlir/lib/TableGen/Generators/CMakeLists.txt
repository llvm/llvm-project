# This library has the same linking constraints as MLIRTableGen: it is used by
# mlir-tblgen (built with DISABLE_LLVM_LINK_LLVM_DYLIB) and must therefore also
# be built with that flag.
llvm_add_library(MLIRTableGenGenerators STATIC
  AttrOrTypeDefGen.cpp
  AttrOrTypeFormatGen.cpp
  BytecodeDialectGen.cpp
  CppGenUtilities.cpp
  DialectGen.cpp
  DialectInterfacesGen.cpp
  DocGenUtilities.cpp
  EnumsGen.cpp
  EnumPythonBindingGen.cpp
  FormatGen.cpp
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

  DISABLE_LLVM_LINK_LLVM_DYLIB

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/TableGen/Generators

  LINK_LIBS PUBLIC
  MLIRTableGen
)
set_target_properties(MLIRTableGenGenerators PROPERTIES FOLDER "MLIR/Tablegenning")

mlir_check_all_link_libraries(MLIRTableGenGenerators)

add_mlir_library_install(MLIRTableGenGenerators)
