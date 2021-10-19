add_mlir_dialect_library(MLIRDLTI
  DLTI.cpp
  Traits.cpp

  DEPENDS
  MLIRDLTIIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDataLayoutInterfaces
  )
