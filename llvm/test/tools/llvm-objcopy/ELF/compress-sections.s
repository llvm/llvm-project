# REQUIRES: x86-registered-target, zlib, zstd

# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
## '*0=none' wins because it is the last. '*0' sections are decompressed (if originally compressed) or kept unchanged (if uncompressed).
## No section is named 'nomatch'. The third option is a no-op.
# RUN: llvm-objcopy a.o out --compress-sections='*0=zlib' --compress-sections '*0=none' --compress-sections 'nomatch=none' 2>&1 | count 0
# RUN: llvm-readelf -S out | FileCheck %s --check-prefix=CHECK1

# CHECK1:      Name           Type          Address     Off      Size     ES Flg Lk Inf Al
# CHECK1:      .text          PROGBITS [[#%x,TEXT:]]    [[#%x,]] [[#%x,]] 00 AX   0   0  4
# CHECK1:      foo0           PROGBITS [[#%x,FOO0:]]    [[#%x,]] [[#%x,]] 00 A    0   0  8
# CHECK1-NEXT: .relafoo0      RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18  I  11   3  8
# CHECK1-NEXT: foo1           PROGBITS [[#%x,FOO1:]]    [[#%x,]] [[#%x,]] 00 A    0   0  8
# CHECK1-NEXT: .relafoo1      RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18  I  11   5  8
# CHECK1:      nonalloc0      PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00      0   0  8
# CHECK1-NEXT: .relanonalloc0 RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18  I  11   7  8
# CHECK1-NEXT: nonalloc1      PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00      0   0  8
# CHECK1-NEXT: .debug_str     PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 01 MS   0   0  1

## Mixing zlib and zstd.
# RUN: llvm-objcopy a.o out2 --compress-sections '*c0=zlib' --compress-sections .debug_str=zstd
# RUN: llvm-readelf -Sr -x nonalloc0 -x .debug_str out2 2>&1 | FileCheck %s --check-prefix=CHECK2
# RUN: llvm-readelf -z -x nonalloc0 -x .debug_str out2 | FileCheck %s --check-prefix=CHECK2DE

# CHECK2:      Name           Type          Address     Off      Size     ES Flg Lk Inf Al
# CHECK2:      .text          PROGBITS [[#%x,TEXT:]]    [[#%x,]] [[#%x,]] 00 AX   0   0  4
# CHECK2:      foo0           PROGBITS [[#%x,FOO0:]]    [[#%x,]] [[#%x,]] 00 A    0   0  8
# CHECK2-NEXT: .relafoo0      RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18  I  11   3  8
# CHECK2-NEXT: foo1           PROGBITS [[#%x,FOO1:]]    [[#%x,]] [[#%x,]] 00 A    0   0  8
# CHECK2-NEXT: .relafoo1      RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18  I  11   5  8
# CHECK2:      nonalloc0      PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00   C  0   0  8
# CHECK2-NEXT: .relanonalloc0 RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18  IC 11   7  8
# CHECK2-NEXT: nonalloc1      PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00      0   0  8
# CHECK2-NEXT: .debug_str     PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 01 MSC  0   0  8

## llvm-readelf -r doesn't support SHF_COMPRESSED SHT_RELA.
# CHECK2: warning: {{.*}}: unable to read relocations from SHT_RELA section with index 8: section [index 8] has an invalid sh_size ([[#]]) which is not a multiple of its sh_entsize (24)

# CHECK2:      Hex dump of section 'nonalloc0':
## zlib with ch_size=0x10
# CHECK2-NEXT: 01000000 00000000 10000000 00000000
# CHECK2-NEXT: 08000000 00000000 {{.*}}
# CHECK2:      Hex dump of section '.debug_str':
## zstd with ch_size=0x38
# CHECK2-NEXT: 02000000 00000000 38000000 00000000
# CHECK2-NEXT: 01000000 00000000 {{.*}}

# CHECK2DE:       Hex dump of section 'nonalloc0':
# CHECK2DE-NEXT:  0x00000000 00000000 00000000 00000000 00000000 ................
# CHECK2DE-EMPTY:
# CHECK2DE-NEXT:  Hex dump of section '.debug_str':
# CHECK2DE-NEXT:  0x00000000 41414141 41414141 41414141 41414141 AAAAAAAAAAAAAAAA

## --decompress-debug-sections takes precedence, even if it is before --compress-sections.
# RUN: llvm-objcopy a.o out3 --decompress-debug-sections --compress-sections .debug_str=zstd
# RUN: llvm-readelf -S out3 | FileCheck %s --check-prefix=CHECK3

# CHECK3:      .debug_str PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 01 MS   0   0  1

# RUN: llvm-objcopy a.o out4 --compress-sections '*0=zlib'
# RUN: llvm-readelf -S out4 | FileCheck %s --check-prefix=CHECK4

# CHECK4:      Name           Type          Address     Off      Size     ES Flg Lk Inf Al
# CHECK4:      .text          PROGBITS [[#%x,TEXT:]]    [[#%x,]] [[#%x,]] 00  AX  0   0  4
# CHECK4:      foo0           PROGBITS [[#%x,FOO0:]]    [[#%x,]] [[#%x,]] 00  AC  0   0  8
# CHECK4-NEXT: .relafoo0      RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18  IC 11   3  8
# CHECK4-NEXT: foo1           PROGBITS [[#%x,FOO1:]]    [[#%x,]] [[#%x,]] 00   A  0   0  8
# CHECK4-NEXT: .relafoo1      RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18   I 11   5  8
# CHECK4:      nonalloc0      PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00   C  0   0  8
# CHECK4-NEXT: .relanonalloc0 RELA     [[#%x,]]         [[#%x,]] [[#%x,]] 18  IC 11   7  8
# CHECK4-NEXT: nonalloc1      PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00      0   0  8
# CHECK4-NEXT: .debug_str     PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 01  MS  0   0  1

## If a section is already compressed, compression request for another format is ignored.
# RUN: llvm-objcopy a.o out5 --compress-sections 'nonalloc0=zlib'
# RUN: llvm-readelf -x nonalloc0 out5 | FileCheck %s --check-prefix=CHECK5
# RUN: llvm-objcopy out5 out5a --compress-sections 'nonalloc0=zstd'
# RUN: cmp out5 out5a

# CHECK5:      Hex dump of section 'nonalloc0':
## zlib with ch_size=0x10
# CHECK5-NEXT: 01000000 00000000 10000000 00000000
# CHECK5-NEXT: 08000000 00000000 {{.*}}

# RUN: not llvm-objcopy --compress-sections=foo a.o out 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR1 --implicit-check-not=error:
# ERR1:      error: --compress-sections: parse error, not 'section-glob=[none|zlib|zstd]'

# RUN: llvm-objcopy --compress-sections 'a[=zlib' a.o out 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR2 --implicit-check-not=error:
# ERR2:      warning: invalid glob pattern, unmatched '['

# RUN: not llvm-objcopy a.o out --compress-sections='.debug*=zlib-gabi' --compress-sections='.debug*=' 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR3 %s
# ERR3:      error: invalid or unsupported --compress-sections format: .debug*=zlib-gabi

# RUN: not llvm-objcopy a.o out --compress-sections='!.debug*=zlib' 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR4 %s
# ERR4:      error: --compress-sections: negative pattern is unsupported

.globl _start
_start:
  ret

.section foo0,"a"
.balign 8
.quad .text-.
.quad .text-.
.section foo1,"a"
.balign 8
.quad .text-.
.quad .text-.
.section nonalloc0,""
.balign 8
.quad .text+1
.quad .text+2
sym0:
.section nonalloc1,""
.balign 8
.quad 42
sym1:

.section .debug_str,"MS",@progbits,1
.Linfo_string0:
  .asciz "AAAAAAAAAAAAAAAAAAAAAAAAAAA"
.Linfo_string1:
  .asciz "BBBBBBBBBBBBBBBBBBBBBBBBBBB"
