set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/lib/Target/RISCV/RISCV.td)
tablegen(LLVM RISCVTargetParserDef.inc -gen-riscv-target-def -I ${CMAKE_SOURCE_DIR}/lib/Target/RISCV/)
add_public_tablegen_target(RISCVTargetParserTableGen)

