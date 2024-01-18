set(LLVM_OPTIONAL_SOURCES
  TestParametricDialect.cpp
)

set(LLVM_TARGET_DEFINITIONS TestParametricInterfaces.td)
mlir_tablegen(TestParametricAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TestParametricAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(TestParametricTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TestParametricTypeInterfaces.cpp.inc -gen-type-interface-defs)
mlir_tablegen(TestParametricOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TestParametricOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTestParametricInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS TestParametricOps.td)
mlir_tablegen(TestParametricAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TestParametricAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTestParametricAttrDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestParametricTypeDefs.td)
mlir_tablegen(TestParametricTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=testparametric)
mlir_tablegen(TestParametricTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=testparametric)
add_public_tablegen_target(MLIRTestParametricTypeDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestParametricOps.td)
mlir_tablegen(TestParametricOps.h.inc -gen-op-decls)
mlir_tablegen(TestParametricOps.cpp.inc -gen-op-defs)
mlir_tablegen(TestParametricOpsDialect.h.inc -gen-dialect-decls -dialect=testparametric)
mlir_tablegen(TestParametricOpsDialect.cpp.inc -gen-dialect-defs -dialect=testparametric)
add_public_tablegen_target(MLIRTestParametricOpsIncGen)

# Exclude testparametrics from libMLIR.so
add_mlir_library(MLIRTestParametricDialect
  TestParametricAttributes.cpp
  TestParametricDialect.cpp
  TestParametricInterfaces.cpp
  TestParametricTypes.cpp
  
  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestParametricAttrDefIncGen
  MLIRTestParametricInterfaceIncGen
  MLIRTestParametricTypeDefIncGen
  MLIRTestParametricOpsIncGen

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
  MLIRReduce
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRTransforms
)

