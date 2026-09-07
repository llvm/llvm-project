add_subdirectory(TransformOps)
add_mlir_dialect_library(MLIRDLTIDialect
  DLTI.cpp
  Traits.cpp

  DEPENDS
  MLIRDLTIAttrsIncGen
  MLIRDLTIIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDataLayoutInterfaces
  )
