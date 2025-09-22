add_mlir_dialect_library(MLIRSMT
  SMTAttributes.cpp
  SMTDialect.cpp
  SMTOps.cpp
  SMTTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SMT

  DEPENDS
  MLIRSMTAttrIncGen
  MLIRSMTEnumsIncGen
  MLIRSMTIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
)

add_dependencies(mlir-headers
  MLIRSMTIncGen
)
