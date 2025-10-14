## Test cfi directives.

# RUN: llvm-mc %s --triple=loongarch32 --mattr=+lasx | FileCheck %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+lasx | FileCheck %s
# RUN: not llvm-mc --triple=loongarch32 --mattr=+lasx --defsym=ERR=1 < %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=CHECK-ERR
# RUN: not llvm-mc --triple=loongarch64 --mattr=+lasx --defsym=ERR=1 < %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=CHECK-ERR

# CHECK: .cfi_startproc
.cfi_startproc
# CHECK-NEXT: .cfi_offset 0, 0
.cfi_offset 0, 0
# CHECK-NEXT: .cfi_offset 9, 8
.cfi_offset 9, 8
# CHECK-NEXT: .cfi_offset 31, 16
.cfi_offset 31, 16
# CHECK-NEXT: .cfi_offset 22, -8
.cfi_offset r22, -8
# CHECK-NEXT: .cfi_offset 22, -8
.cfi_offset $r22, -8
# CHECK-NEXT: .cfi_offset 22, -8
.cfi_offset fp, -8
# CHECK-NEXT: .cfi_offset 22, -8
.cfi_offset $fp, -8
# CHECK-NEXT: .cfi_offset 42, 8
.cfi_offset f10, 8
# CHECK-NEXT: .cfi_offset 56, 8
.cfi_offset fs0, 8
# CHECK-NEXT: .cfi_endproc
.cfi_endproc

.ifdef ERR
.cfi_startproc
# CHECK-ERR: :[[#@LINE+1]]:13: error: invalid register name
.cfi_offset -22, -8
# CHECK-ERR: :[[#@LINE+1]]:13: error: invalid register name
.cfi_offset lr, -8
# CHECK-ERR: :[[#@LINE+1]]:13: error: invalid register name
.cfi_offset r32, -8
# CHECK-ERR: :[[#@LINE+1]]:14: error: invalid register name
.cfi_offset $r32, -8
# CHECK-ERR: :[[#@LINE+1]]:14: error: invalid register name
.cfi_offset $22, -8
# CHECK-ERR: :[[#@LINE+1]]:16: error: invalid register name
.cfi_offset vr0, 8
# CHECK-ERR: :[[#@LINE+1]]:16: error: invalid register name
.cfi_offset xr0, 8
.cfi_endproc
.endif
