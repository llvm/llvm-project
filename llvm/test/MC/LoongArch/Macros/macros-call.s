# RUN: llvm-mc --triple=loongarch64 %s | FileCheck %s

call36 sym_call
# CHECK:      pcaddu18i $ra, %call36(sym_call)
# CHECK-NEXT: jirl $ra, $ra, 0

tail36 $t0, sym_tail
# CHECK:      pcaddu18i $t0, %call36(sym_tail)
# CHECK-NEXT: jr $t0
