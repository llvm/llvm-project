add_mlir_library(MLIRPDLLODS
  Constraint.cpp
  Context.cpp
  Dialect.cpp
  Operation.cpp

  # The PDLL ODS model defines its own Dialect and Operation in mlir::pdll::ods,
  # which collide with the mlir:: ones in the MLIRIR PCH. This library only
  # depends on MLIRSupport, so it has no use for that PCH.
  DISABLE_PCH_REUSE

  LINK_LIBS PUBLIC
  MLIRSupport
  )
