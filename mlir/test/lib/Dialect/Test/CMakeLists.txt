set(LLVM_OPTIONAL_SOURCES
  TestDialect.cpp
  TestPatterns.cpp
  TestTraits.cpp
)

set(LLVM_TARGET_DEFINITIONS TestInterfaces.td)
mlir_tablegen(TestTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TestTypeInterfaces.cpp.inc -gen-type-interface-defs)
mlir_tablegen(TestOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TestOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTestInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS TestTypeDefs.td)
mlir_tablegen(TestTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(TestTypeDefs.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRTestDefIncGen)


set(LLVM_TARGET_DEFINITIONS TestOps.td)
mlir_tablegen(TestOps.h.inc -gen-op-decls)
mlir_tablegen(TestOps.cpp.inc -gen-op-defs)
mlir_tablegen(TestOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TestOpEnums.h.inc -gen-enum-decls)
mlir_tablegen(TestOpEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TestOpStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(TestOpStructs.cpp.inc -gen-struct-attr-defs)
mlir_tablegen(TestPatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRTestOpsIncGen)

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestDialect
  TestDialect.cpp
  TestInterfaces.cpp
  TestPatterns.cpp
  TestTraits.cpp
  TestTypes.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestInterfaceIncGen
  MLIRTestDefIncGen
  MLIRTestOpsIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRDerivedAttributeOpInterface
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgTransforms
  MLIRPass
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRTransformUtils
  MLIRTransforms
)
