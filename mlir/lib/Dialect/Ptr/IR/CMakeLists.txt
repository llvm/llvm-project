add_mlir_dialect_library(
  MLIRPtrDialect
  PtrAttrs.cpp
  PtrTypes.cpp
  PtrDialect.cpp

  DEPENDS
  MLIRPtrOpsAttributesIncGen
  MLIRPtrOpsIncGen
  MLIRPtrOpsEnumsGen
  MLIRPtrMemorySpaceInterfacesIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRDataLayoutInterfaces
  MLIRMemorySlotInterfaces
)
