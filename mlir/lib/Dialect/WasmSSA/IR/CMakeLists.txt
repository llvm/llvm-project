add_mlir_dialect_library(MLIRWasmSSADialect
  WasmSSAOps.cpp
  WasmSSADialect.cpp
  WasmSSAInterfaces.cpp
  WasmSSATypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/WasmSSA

  DEPENDS
  MLIRWasmSSAOpsIncGen
  MLIRWasmSSAInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRCastInterfaces
  MLIRDataLayoutInterfaces
  MLIRDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSupport

  PRIVATE
  MLIRFunctionInterfaces
  )
