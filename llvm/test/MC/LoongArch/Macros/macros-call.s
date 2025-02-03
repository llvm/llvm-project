# RUN: llvm-mc --triple=loongarch64 %s | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.relax
# RUN: llvm-readobj -r %t.relax | FileCheck %s --check-prefixes=RELOC,RELAX

# RELOC:      Relocations [
# RELOC-NEXT:   Section ({{.*}}) .rela.text {

call36 sym_call
# CHECK:      pcaddu18i $ra, %call36(sym_call)
# CHECK-NEXT: jirl $ra, $ra, 0

# RELOC-NEXT: R_LARCH_CALL36 sym_call 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

tail36 $t0, sym_tail
# CHECK:      pcaddu18i $t0, %call36(sym_tail)
# CHECK-NEXT: jr $t0

# RELOC-NEXT: R_LARCH_CALL36 sym_tail 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0


# RELOC-NEXT:   }
# RELOC-NEXT: ]
