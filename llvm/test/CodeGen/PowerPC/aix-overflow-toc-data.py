# UNSUPPORTED: expensive_checks, debug

# RUN: %python %s > %t.ll
# RUN: llc -mtriple powerpc-ibm-aix-xcoff -code-model=small -mcpu=pwr7 -mattr=-altivec -O0 < %t.ll | \
# RUN:   FileCheck --check-prefix=ASM32 %s

# RUN: llc -mtriple powerpc64-ibm-aix-xcoff -code-model=small -mcpu=pwr7 -mattr=-altivec -O0 < %t.ll | \
# RUN:   FileCheck --check-prefix=ASM64 %s

# RUN: llc -mtriple powerpc-ibm-aix-xcoff -code-model=small -mcpu=pwr7 -mattr=-altivec -O0 \
# RUN:     -filetype=obj -o %t.o < %t.ll
# RUN: llvm-objdump --no-print-imm-hex -D -r --symbol-description %t.o | FileCheck -D#NFA=2 --check-prefix=DIS32 %s

# RUN: llc -mtriple powerpc64-ibm-aix-xcoff -code-model=small -mcpu=pwr7 -mattr=-altivec -O0 \
# RUN:     -filetype=obj -o %t.o < %t.ll
# RUN: llvm-objdump --no-print-imm-hex -D -r --symbol-description %t.o | FileCheck -D#NFA=2 --check-prefix=DIS64 %s

numentries = 8195
for x in range(0, numentries):
    print("@a%d = global i32 0, align 4 #0" % (x))

print("define void @foo() {")
print("entry:")
for x in range(0, numentries):
    print("store i32 1, i32* @a%d, align 4" % (x))
print("ret void")
print("}")

print('attributes #0 = { "toc-data" }')

# 32-bit assembly check
# ASM32:  la 4, a0[TD](2)
# ASM32:  la 4, a1[TD](2)

# ASM32:  la 4, a8191[TD](2)
# ASM32:  la 4, a8192[TD](2)
# ASM32:  la 4, a8193[TD](2)

# 64-bit assembly check
# ASM64:  la 4, a0[TD](2)
# ASM64:  la 4, a1[TD](2)

# ASM64:  la 4, a8191[TD](2)
# ASM64:  la 4, a8192[TD](2)
# ASM64:  la 4, a8193[TD](2)

# DIS32:    fffc: 38 82 7f fc   addi 4, 2, 32764
# DIS32:      0000fffe:  R_TOC  (idx: [[#NFA+16391]]) a8191[TD]
# DIS32:   10004: 38 82 80 00   addi 4, 2, -32768
# DIS32:      00010006:  R_TOC  (idx: [[#NFA+16393]]) a8192[TD]
# DIS32:   1000c: 38 82 80 04   addi 4, 2, -32764
# DIS32:      0001000e:  R_TOC  (idx: [[#NFA+16395]]) a8193[TD]

# DIS64:    fffc: 38 82 7f fc   addi 4, 2, 32764
# DIS64:      0000fffe:  R_TOC  (idx: [[#NFA+16391]]) a8191[TD]
# DIS64:   10004: 38 82 80 00   addi 4, 2, -32768
# DIS64:      00010006:  R_TOC  (idx: [[#NFA+16393]]) a8192[TD]
# DIS64:   1000c: 38 82 80 04   addi 4, 2, -32764
# DIS64:      0001000e:  R_TOC  (idx: [[#NFA+16395]]) a8193[TD]
