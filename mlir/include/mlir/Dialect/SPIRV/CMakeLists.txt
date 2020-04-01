add_mlir_dialect(SPIRVOps spv)
add_mlir_doc(SPIRVOps -gen-op-doc SPIRVOps Dialects/)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVEnums.h.inc -gen-enum-decls)
mlir_tablegen(SPIRVEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSPIRVEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVEnumAvailability.h.inc -gen-spirv-enum-avail-decls)
mlir_tablegen(SPIRVEnumAvailability.cpp.inc -gen-spirv-enum-avail-defs)
mlir_tablegen(SPIRVCapabilityImplication.inc -gen-spirv-capability-implication)
add_public_tablegen_target(MLIRSPIRVEnumAvailabilityIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVAvailability.h.inc -gen-avail-interface-decls)
mlir_tablegen(SPIRVAvailability.cpp.inc -gen-avail-interface-defs)
mlir_tablegen(SPIRVOpAvailabilityImpl.inc -gen-spirv-avail-impls)
add_public_tablegen_target(MLIRSPIRVAvailabilityIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVSerialization.inc -gen-spirv-serialization)
add_public_tablegen_target(MLIRSPIRVSerializationGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVOpUtils.inc -gen-spirv-op-utils)
add_public_tablegen_target(MLIRSPIRVOpUtilsGen)

set(LLVM_TARGET_DEFINITIONS TargetAndABI.td)
mlir_tablegen(TargetAndABI.h.inc -gen-struct-attr-decls)
mlir_tablegen(TargetAndABI.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRSPIRVTargetAndABIIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRSPIRVPassIncGen)
