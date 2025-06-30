add_mlir_dialect_library(MLIRWasmSSADialect
  WebAssemblySSAOps.cpp
  WebAssemblySSADialect.cpp
  WebAssemblySSAInterfaces.cpp
  WebAssemblySSATypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/WebAssemblySSA

  DEPENDS
  MLIRWebAssemblySSAOpsIncGen
  MLIRWebAssemblySSAInterfacesIncGen

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
