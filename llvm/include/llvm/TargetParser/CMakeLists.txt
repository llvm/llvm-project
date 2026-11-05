set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/AArch64/AArch64TargetParserDef.td)
llvm_tablegen(AArch64TargetParserDef.inc -gen-arm-target-def EXTRA_INCLUDES ${PROJECT_SOURCE_DIR}/lib/Target/AArch64)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/RISCV/RISCV.td)
llvm_tablegen(RISCVTargetParserDef.inc -gen-riscv-target-def EXTRA_INCLUDES ${PROJECT_SOURCE_DIR}/lib/Target/RISCV)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/PowerPC/PPCTargetParserDef.td)
llvm_tablegen(PPCGenTargetFeatures.inc -gen-target-features EXTRA_INCLUDES ${PROJECT_SOURCE_DIR}/lib/Target/PowerPC)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU/R600TargetParserDef.td)
llvm_tablegen(R600TargetParserDef.inc -gen-amdgpu-target-def EXTRA_INCLUDES ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU/AMDGPUTargetParserDef.td)
llvm_tablegen(AMDGPUTargetParserDef.inc -gen-amdgpu-target-def EXTRA_INCLUDES ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU)

# This covers all of the tablegen calls above.
add_public_tablegen_target(target_parser_gen)
