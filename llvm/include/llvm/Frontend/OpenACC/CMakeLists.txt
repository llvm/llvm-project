set(LLVM_TARGET_DEFINITIONS ACC.td)
tablegen(LLVM ACC.h.inc --gen-directive-decl)
tablegen(LLVM ACC.cpp.inc --gen-directive-gen)
add_public_tablegen_target(acc_gen)
