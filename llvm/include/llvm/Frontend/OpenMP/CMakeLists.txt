set(LLVM_TARGET_DEFINITIONS OMP.td)
llvm_tablegen(OMP.h.inc --gen-directive-decl)
llvm_tablegen(OMP.inc --gen-directive-impl)
add_public_tablegen_target(omp_gen)
