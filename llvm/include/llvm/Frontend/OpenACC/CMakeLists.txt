set(LLVM_TARGET_DEFINITIONS ACC.td)
llvm_tablegen(ACC.h.inc --gen-directive-decl)
llvm_tablegen(ACC.inc --gen-directive-impl)
add_public_tablegen_target(acc_gen)
