add_mlir_library(MLIRPDLLAST
  Context.cpp
  Diagnostic.cpp
  NodePrinter.cpp
  Nodes.cpp
  Types.cpp

  # The PDLL AST defines its own Type, TupleType and InFlightDiagnostic in
  # mlir::pdll::ast, which collide with the mlir:: ones in the MLIRIR PCH.
  # This library only depends on MLIRSupport, so it has no use for that PCH.
  DISABLE_PCH_REUSE

  LINK_LIBS PUBLIC
  MLIRPDLLODS
  MLIRSupport
  )
