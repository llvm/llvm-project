set(LLVM_OPTIONAL_SOURCES
  TestDialect.cpp
  TestPatterns.cpp
  TestTraits.cpp
)

set(LLVM_TARGET_DEFINITIONS TestInterfaces.td)
mlir_tablegen(TestAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TestAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(TestTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TestTypeInterfaces.cpp.inc -gen-type-interface-defs)
mlir_tablegen(TestOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TestOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTestInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS TestAttrDefs.td)
mlir_tablegen(TestAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TestAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTestAttrDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestTypeDefs.td)
mlir_tablegen(TestTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=test)
mlir_tablegen(TestTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=test)
add_public_tablegen_target(MLIRTestTypeDefIncGen)


set(LLVM_TARGET_DEFINITIONS TestOps.td)
mlir_tablegen(TestOps.h.inc -gen-op-decls)
mlir_tablegen(TestOps.cpp.inc -gen-op-defs)
mlir_tablegen(TestOpsDialect.h.inc -gen-dialect-decls -dialect=test)
mlir_tablegen(TestOpsDialect.cpp.inc -gen-dialect-defs -dialect=test)
mlir_tablegen(TestOpEnums.h.inc -gen-enum-decls)
mlir_tablegen(TestOpEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TestOpStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(TestOpStructs.cpp.inc -gen-struct-attr-defs)
mlir_tablegen(TestPatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRTestOpsIncGen)

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestDialect
  TestAttributes.cpp
  TestDialect.cpp
  TestInterfaces.cpp
  TestPatterns.cpp
  TestTraits.cpp
  TestTypes.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestAttrDefIncGen
  MLIRTestInterfaceIncGen
  MLIRTestTypeDefIncGen
  MLIRTestOpsIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDerivedAttributeOpInterface
  MLIRDialect
  MLIRDLTI
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgTransforms
  MLIRPass
  MLIRReduce
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRTensor
  MLIRTransformUtils
  MLIRTransforms
)
