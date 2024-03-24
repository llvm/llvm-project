set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/lib/Target/ARM/ARM.td)
tablegen(LLVM JITLinkAArch32.inc -gen-jitlink-aarch32-instr-info -I ${PROJECT_SOURCE_DIR}/lib/Target/ARM/)
add_public_tablegen_target(JITLinkAArch32TableGen)
