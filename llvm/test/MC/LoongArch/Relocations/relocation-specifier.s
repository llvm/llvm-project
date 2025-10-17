# RUN: llvm-mc --filetype=obj --triple=loongarch32 %s -o %t-la32
# RUN: llvm-readelf -rs %t-la32 | FileCheck %s --check-prefixes=CHECK,RELOC32
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t-la64
# RUN: llvm-readelf -rs %t-la64 | FileCheck %s --check-prefixes=CHECK,RELOC64

## This test is similar to test/MC/CSKY/relocation-specifier.s.

# RELOC32: '.rela.data'
# RELOC32: R_LARCH_32 00000000 local

# RELOC64: '.rela.data'
# RELOC64: R_LARCH_32 0000000000000000 local

# CHECK: TLS GLOBAL DEFAULT UND gd
# CHECK: TLS GLOBAL DEFAULT UND ld
# CHECK: TLS GLOBAL DEFAULT UND ie
# CHECK: TLS GLOBAL DEFAULT UND le

pcalau12i $t1, %gd_pc_hi20(gd)
pcalau12i $t1, %ld_pc_hi20(ld)
pcalau12i $t1, %ie_pc_hi20(ie)
lu12i.w $t1, %le_hi20_r(le)

.data
local:
.long local
