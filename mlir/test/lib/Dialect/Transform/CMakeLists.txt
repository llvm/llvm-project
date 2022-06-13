set(LLVM_TARGET_DEFINITIONS TestTransformDialectExtension.td)
mlir_tablegen(TestTransformDialectExtension.h.inc -gen-op-decls)
mlir_tablegen(TestTransformDialectExtension.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTestTransformDialectExtensionIncGen)

add_mlir_library(MLIRTestTransformDialect
  TestTransformDialectExtension.cpp
  TestTransformDialectInterpreter.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestTransformDialectExtensionIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRPDLDialect
  MLIRTransformDialect
)
