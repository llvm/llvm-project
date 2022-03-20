# UNSUPPORTED: expensive_checks, debug

# RUN: %python %s > %t.ll
# RUN: llc -mtriple powerpc-ibm-aix-xcoff -code-model=small -data-sections=false -mcpu=pwr4 -mattr=-altivec -O0 < %t.ll | \
# RUN:   FileCheck --check-prefix=ASM32 %s

# RUN: llc -mtriple powerpc64-ibm-aix-xcoff -code-model=small -data-sections=false -mcpu=pwr4 -mattr=-altivec -O0 < %t.ll | \
# RUN:   FileCheck --check-prefix=ASM64 %s

# RUN: llc -mtriple powerpc-ibm-aix-xcoff -code-model=small -data-sections=false -mcpu=pwr4 -mattr=-altivec -O0 \
# RUN:     -filetype=obj -o %t.o < %t.ll
# RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS32 %s

## FIXME: currently only fileHeader and sectionHeaders are supported in XCOFF64.

numentries = 12290
for x in range(0, numentries):
    print("@a%d = global i32 0, align 4" % (x))

print("define void @foo() {")
print("entry:")
for x in range(0, numentries):
    print("store i32 1, i32* @a%d, align 4" % (x))
print("ret void")
print("}")

# 32-bit assembly check
# ASM32:  lwz 4, L..C0(2) # @a0
# ASM32:  lwz 4, L..C1(2) # @a1

# ASM32:  lwz 4, L..C8191(2) # @a8191
# ASM32:  lwz 4, L..C8192-65536(2) # @a8192
# ASM32:  lwz 4, L..C8193-65536(2) # @a8193

# ASM32:  lwz 4, L..C12288-65536(2) # @a12288
# ASM32:  lwz 4, L..C12289-65536(2) # @a12289

# 64-bit assembly check
# ASM64:  ld 4, L..C0(2) # @a0
# ASM64:  ld 4, L..C1(2) # @a1

# ASM64:  ld 4, L..C4095(2) # @a4095
# ASM64:  ld 4, L..C4096-65536(2) # @a4096
# ASM64:  ld 4, L..C4097-65536(2) # @a4097

# ASM64:  ld 4, L..C12287-65536(2) # @a12287
# ASM64:  ld 4, L..C12288-131072(2) # @a12288
# ASM64:  ld 4, L..C12289-131072(2) # @a12289

# DIS32:   0: 80 82 00 00   lwz 4, 0(2)
# DIS32:  00000002:  R_TOC  (idx: 24591) a0[TC]
# DIS32:   c: 80 82 00 04   lwz 4, 4(2)
# DIS32:  0000000e:  R_TOC  (idx: 24593) a1[TC]

# DIS32:    fffc: 80 82 7f fc   lwz 4, 32764(2)
# DIS32:      0000fffe:  R_TOC  (idx: 40973) a8191[TC]
# DIS32:   10004: 80 82 80 00   lwz 4, -32768(2)
# DIS32:      00010006:  R_TOC  (idx: 40975) a8192[TC]
# DIS32:   1000c: 80 82 80 04   lwz 4, -32764(2)
# DIS32:      0001000e:  R_TOC  (idx: 40977) a8193[TC]

# DIS32:   18004: 80 82 c0 00   lwz 4, -16384(2)
# DIS32:      00018006:  R_TOC  (idx: 49167) a12288[TC]
# DIS32:   1800c: 80 82 c0 04   lwz 4, -16380(2)
# DIS32:      0001800e:  R_TOC  (idx: 49169) a12289[TC]
