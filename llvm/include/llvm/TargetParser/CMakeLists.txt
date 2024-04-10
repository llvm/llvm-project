set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/ARM/ARM.td)
tablegen(LLVM ARMTargetParserDef.inc -gen-arm-target-def -I ${PROJECT_SOURCE_DIR}/lib/Target/ARM/)
add_public_tablegen_target(ARMTargetParserTableGen)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/RISCV/RISCV.td)
tablegen(LLVM RISCVTargetParserDef.inc -gen-riscv-target-def -I ${PROJECT_SOURCE_DIR}/lib/Target/RISCV/)
add_public_tablegen_target(RISCVTargetParserTableGen)

