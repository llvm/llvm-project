# RUN: llvm-mc --triple=loongarch32 %s | FileCheck %s --check-prefixes=CHECK,LA32
# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=-relax %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefixes=RELOC,LA32-RELOC
# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax %s -o %t.relax
# RUN: llvm-readobj -r %t.relax | FileCheck %s --check-prefixes=RELOC,RELAX,LA32-RELOC
# RUN: llvm-mc --triple=loongarch64 %s --defsym=LA64=1 | FileCheck %s --check-prefixes=CHECK,LA64
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s -o %t --defsym=LA64=1
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefixes=RELOC,LA64-RELOC
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.relax --defsym=LA64=1
# RUN: llvm-readobj -r %t.relax | FileCheck %s --check-prefixes=RELOC,RELAX,LA64-RELOC,LA64-RELAX

# RELOC:      Relocations [
# RELOC-NEXT:   Section ({{.*}}) .rela.text {

call sym_call
# LA32:      pcaddu12i $ra, %call30(sym_call)
# LA32-NEXT: jirl $ra, $ra, 0
# LA64:      pcaddu18i $ra, %call36(sym_call)
# LA64-NEXT: jirl $ra, $ra, 0

# LA32-RELOC-NEXT: R_LARCH_CALL30 sym_call 0x0
# LA64-RELOC-NEXT: R_LARCH_CALL36 sym_call 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

tail $t0, sym_tail
# LA32:      pcaddu12i $t0, %call30(sym_tail)
# LA32-NEXT: jr $t0
# LA64:      pcaddu18i $t0, %call36(sym_tail)
# LA64-NEXT: jr $t0

# LA32-RELOC-NEXT: R_LARCH_CALL30 sym_tail 0x0
# LA64-RELOC-NEXT: R_LARCH_CALL36 sym_tail 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

call30 sym_call
# CHECK:      pcaddu12i $ra, %call30(sym_call)
# CHECK-NEXT: jirl $ra, $ra, 0

# RELOC-NEXT: R_LARCH_CALL30 sym_call 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

tail30 $t0, sym_tail
# CHECK:      pcaddu12i $t0, %call30(sym_tail)
# CHECK-NEXT: jr $t0

# RELOC-NEXT: R_LARCH_CALL30 sym_tail 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

.ifdef LA64

call36 sym_call
# LA64:      pcaddu18i $ra, %call36(sym_call)
# LA64-NEXT: jirl $ra, $ra, 0

# LA64-RELOC-NEXT: R_LARCH_CALL36 sym_call 0x0
# LA64-RELAX-NEXT: R_LARCH_RELAX - 0x0

tail36 $t0, sym_tail
# LA64:      pcaddu18i $t0, %call36(sym_tail)
# LA64-NEXT: jr $t0

# LA64-RELOC-NEXT: R_LARCH_CALL36 sym_tail 0x0
# LA64-RELAX-NEXT: R_LARCH_RELAX - 0x0

.endif

# RELOC-NEXT:   }
# RELOC-NEXT: ]
