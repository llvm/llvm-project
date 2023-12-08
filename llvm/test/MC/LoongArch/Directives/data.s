## Test data directives.
# RUN: llvm-mc --triple=loongarch32 < %s \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc --triple=loongarch64 < %s \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc --triple=loongarch32 --filetype=obj < %s | llvm-objdump -s - \
# RUN:     | FileCheck --check-prefix=CHECK-DATA %s
# RUN: llvm-mc --triple=loongarch64 --filetype=obj < %s | llvm-objdump -s - \
# RUN:     | FileCheck --check-prefix=CHECK-DATA %s
# RUN: not llvm-mc --triple=loongarch32 --defsym=ERR=1 < %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=CHECK-ERR
# RUN: not llvm-mc --triple=loongarch64 --defsym=ERR=1 < %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=CHECK-ERR

.data

# CHECK-ASM:      .byte 0
# CHECK-ASM-NEXT: .byte 1
# CHECK-ASM-NEXT: .byte 171
# CHECK-ASM-NEXT: .byte 255
# CHECK-DATA: Contents of section .data:
# CHECK-DATA-NEXT: 0000 0001abff 0100ffff 0100ffff 0100ffff
.byte 0
.byte 1
.byte 0xab
.byte 0xff

# CHECK-ASM:      .half 1
# CHECK-ASM-NEXT: .half 65535
.half 0x1
.half 0xffff

# CHECK-ASM:      .half 1
# CHECK-ASM-NEXT: .half 65535
.2byte 0x1
.2byte 0xffff

# CHECK-ASM:      .half 1
# CHECK-ASM-NEXT: .half 65535
.short 0x1
.short 0xffff

# CHECK-ASM:      .half 0
# CHECK-ASM-NEXT: .half 1
# CHECK-ASM-NEXT: .half 4660
# CHECK-ASM-NEXT: .half 65535
# CHECK-DATA-NEXT: 0010 00000100 3412ffff 01000000 ffffffff
.hword 0
.hword 0x1
.hword 0x1234
.hword 0xffff

# CHECK-ASM:      .word 1
# CHECK-ASM-NEXT: .word 4294967295
.word 0x1
.word 0xffffffff

# CHECK-ASM:      .word 1
# CHECK-ASM-NEXT: .word 4294967295
# CHECK-DATA-NEXT: 0020 01000000 ffffffff 01000000 ffffffff
.long 0x1
.long 0xffffffff

# CHECK-ASM:      .word 1
# CHECK-ASM-NEXT: .word 4294967295
.4byte 0x1
.4byte 0xffffffff

# CHECK-ASM:      .dword 1
# CHECK-ASM-NEXT: .dword 1234605616436508552
# CHECK-DATA-NEXT: 0030 01000000 00000000 88776655 44332211
.dword 0x1
.dword 0x1122334455667788

# CHECK-ASM:      .dword 1
# CHECK-ASM-NEXT: .dword 1234605616436508552
# CHECK-DATA-NEXT: 0040 01000000 00000000 88776655 44332211
.8byte 0x1
.8byte 0x1122334455667788

.ifdef ERR
# CHECK-ERR: :[[#@LINE+1]]:7: error: out of range literal value
.byte 0xffa
# CHECK-ERR: :[[#@LINE+1]]:7: error: out of range literal value
.half 0xffffa
# CHECK-ERR: :[[#@LINE+1]]:8: error: out of range literal value
.short 0xffffa
# CHECK-ERR: :[[#@LINE+1]]:8: error: out of range literal value
.hword 0xffffa
# CHECK-ERR: :[[#@LINE+1]]:8: error: out of range literal value
.2byte 0xffffa
# CHECK-ERR: :[[#@LINE+1]]:7: error: out of range literal value
.word 0xffffffffa
# CHECK-ERR: :[[#@LINE+1]]:7: error: out of range literal value
.long 0xffffffffa
# CHECK-ERR: :[[#@LINE+1]]:8: error: out of range literal value
.4byte 0xffffffffa
# CHECK-ERR: :[[#@LINE+1]]:8: error: literal value out of range for directive
.dword 0xffffffffffffffffa
# CHECK-ERR: :[[#@LINE+1]]:8: error: literal value out of range for directive
.8byte 0xffffffffffffffffa
.endif
