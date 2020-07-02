set(LLVM_TARGET_DEFINITIONS OMP.td)
tablegen(LLVM OMP.h.inc --gen-directive-decls)
add_public_tablegen_target(omp_gen)
