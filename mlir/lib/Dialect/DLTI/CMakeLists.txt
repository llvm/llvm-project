add_subdirectory(TransformOps)
add_mlir_dialect_library(MLIRDLTIDialect
  DLTI.cpp
  Traits.cpp

  DEPENDS
  MLIRDLTIIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDataLayoutInterfaces
  )
