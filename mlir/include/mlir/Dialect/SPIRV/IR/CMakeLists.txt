add_mlir_dialect(SPIRVOps spirv)
add_mlir_doc(SPIRVOps SPIRVOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVEnums.h.inc -gen-enum-decls)
mlir_tablegen(SPIRVEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRSPIRVEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVEnumAvailability.h.inc -gen-spirv-enum-avail-decls)
mlir_tablegen(SPIRVEnumAvailability.cpp.inc -gen-spirv-enum-avail-defs)
mlir_tablegen(SPIRVCapabilityImplication.inc -gen-spirv-capability-implication)
add_mlir_dialect_tablegen_target(MLIRSPIRVEnumAvailabilityIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVAvailability.h.inc -gen-avail-interface-decls)
mlir_tablegen(SPIRVAvailability.cpp.inc -gen-avail-interface-defs)
mlir_tablegen(SPIRVOpAvailabilityImpl.inc -gen-spirv-avail-impls)
add_mlir_dialect_tablegen_target(MLIRSPIRVAvailabilityIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVSerialization.inc -gen-spirv-serialization)
add_mlir_dialect_tablegen_target(MLIRSPIRVSerializationGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVAttrUtils.inc -gen-spirv-attr-utils)
add_mlir_dialect_tablegen_target(MLIRSPIRVAttrUtilsGen)

set(LLVM_TARGET_DEFINITIONS SPIRVAttributes.td)
mlir_tablegen(SPIRVAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SPIRVAttributes.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIRSPIRVAttributeIncGen)
