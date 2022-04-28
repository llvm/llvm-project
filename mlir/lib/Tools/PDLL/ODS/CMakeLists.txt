add_mlir_library(MLIRPDLLODS
  Constraint.cpp
  Context.cpp
  Dialect.cpp
  Operation.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
  )
