add_mlir_interface(CallInterfaces)
add_mlir_interface(CastInterfaces)
add_mlir_interface(ControlFlowInterfaces)
add_mlir_interface(DerivedAttributeOpInterface)
add_mlir_interface(DestinationStyleOpInterface)
add_mlir_interface(FunctionInterfaces)
add_mlir_interface(IndexingMapOpInterface)
add_mlir_interface(InferIntRangeInterface)
add_mlir_interface(InferStridedMetadataInterface)
add_mlir_interface(InferTypeOpInterface)
add_mlir_interface(LoopLikeInterface)
add_mlir_interface(MemOpInterfaces)
add_mlir_interface(ParallelCombiningOpInterface)
add_mlir_interface(RuntimeVerifiableOpInterface)
add_mlir_interface(ShapedOpInterfaces)
add_mlir_interface(SideEffectInterfaces)
add_mlir_interface(SubsetOpInterface)
add_mlir_interface(TilingInterface)
add_mlir_interface(ValueBoundsOpInterface)
add_mlir_interface(VectorInterfaces)
add_mlir_interface(ViewLikeInterface)

set(LLVM_TARGET_DEFINITIONS MemorySlotInterfaces.td)
mlir_tablegen(MemorySlotOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(MemorySlotOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(MemorySlotTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(MemorySlotTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_mlir_generic_tablegen_target(MLIRMemorySlotInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS DataLayoutInterfaces.td)
mlir_tablegen(DataLayoutAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(DataLayoutAttrInterface.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(DataLayoutOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(DataLayoutOpInterface.cpp.inc -gen-op-interface-defs)
mlir_tablegen(DataLayoutTypeInterface.h.inc -gen-type-interface-decls)
mlir_tablegen(DataLayoutTypeInterface.cpp.inc -gen-type-interface-defs)
add_mlir_generic_tablegen_target(MLIRDataLayoutInterfacesIncGen)

add_mlir_doc(DataLayoutInterfaces
  DataLayoutAttrInterface
  Interfaces/
  -gen-attr-interface-docs)

add_mlir_doc(DataLayoutInterfaces
  DataLayoutTypeInterface
  Interfaces/
  -gen-type-interface-docs)

add_mlir_doc(DataLayoutInterfaces
  DataLayoutOpInterface
  Interfaces/
  -gen-op-interface-docs)
