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
  MLIRViewLikeInterface
)

set_target_properties(obj.MLIRPtrDialect PROPERTIES UNITY_BUILD_BATCH_SIZE 1)
