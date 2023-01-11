set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/RISCV/RISCV.td)
tablegen(LLVM RISCVTargetParserDef.inc -gen-riscv-target-def -I ${PROJECT_SOURCE_DIR}/lib/Target/RISCV/)
add_public_tablegen_target(RISCVTargetParserTableGen)
