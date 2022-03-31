add_mlir_library(MLIRPDLLODS
  Context.cpp
  Dialect.cpp
  Operation.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
  )
