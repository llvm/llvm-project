set(LLVM_OPTIONAL_SOURCES
  TestDialect.cpp
  TestPatterns.cpp
  TestTraits.cpp
  TestFromLLVMIRTranslation.cpp
  TestToLLVMIRTranslation.cpp
)

set(LLVM_TARGET_DEFINITIONS TestInterfaces.td)
mlir_tablegen(TestAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TestAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(TestTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TestTypeInterfaces.cpp.inc -gen-type-interface-defs)
mlir_tablegen(TestOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TestOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTestInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS TestOps.td)
mlir_tablegen(TestAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=test)
mlir_tablegen(TestAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=test)
add_public_tablegen_target(MLIRTestAttrDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestTypeDefs.td)
mlir_tablegen(TestTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=test)
mlir_tablegen(TestTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=test)
add_public_tablegen_target(MLIRTestTypeDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestEnumDefs.td)
mlir_tablegen(TestOpEnums.h.inc -gen-enum-decls)
mlir_tablegen(TestOpEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTestEnumDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestOps.td)
mlir_tablegen(TestOpsDialect.h.inc -gen-dialect-decls -dialect=test)
mlir_tablegen(TestOpsDialect.cpp.inc -gen-dialect-defs -dialect=test)
mlir_tablegen(TestPatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRTestOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TestOpsSyntax.td)
mlir_tablegen(TestOpsSyntax.h.inc -gen-op-decls)
mlir_tablegen(TestOpsSyntax.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTestOpsSyntaxIncGen)

add_sharded_ops(TestOps 20)

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestDialect
  TestAttributes.cpp
  TestDialect.cpp
  TestFormatUtils.cpp
  TestInterfaces.cpp
  TestOpDefs.cpp
  TestOps.cpp
  TestPatterns.cpp
  TestTraits.cpp
  TestTypes.cpp
  TestOpsSyntax.cpp
  TestDialectInterfaces.cpp
  ${SHARDED_SRCS}

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestAttrDefIncGen
  MLIRTestEnumDefIncGen
  MLIRTestInterfaceIncGen
  MLIRTestTypeDefIncGen
  MLIRTestOpsIncGen
  MLIRTestOpsSyntaxIncGen
  MLIRTestOpsShardGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDerivedAttributeOpInterface
  MLIRDestinationStyleOpInterface
  MLIRDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRFuncTransforms
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRPass
  MLIRPolynomialDialect
  MLIRReduce
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRTransforms
  MLIRValueBoundsOpInterface
)

add_mlir_translation_library(MLIRTestFromLLVMIRTranslation
  TestFromLLVMIRTranslation.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRTestDialect
  MLIRSupport
  MLIRTargetLLVMIRImport
  MLIRLLVMIRToLLVMTranslation
)

add_mlir_translation_library(MLIRTestToLLVMIRTranslation
  TestToLLVMIRTranslation.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRTestDialect
  MLIRSupport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)
