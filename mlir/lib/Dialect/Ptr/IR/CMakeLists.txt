set(LLVM_OPTIONAL_SOURCES
  MemorySpaceInterfaces.cpp
  PtrAttrs.cpp
  PtrTypes.cpp
  PtrDialect.cpp
)

add_mlir_dialect_library(
  MLIRPtrMemorySpaceInterfaces
  MemorySpaceInterfaces.cpp

  DEPENDS
  MLIRPtrOpsEnumsGen
  MLIRPtrMemorySpaceInterfacesIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
)

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
  MLIRPtrMemorySpaceInterfaces
)
