# RUN: llvm-mc -triple=aarch64 -filetype=obj %s | llvm-readobj -r - | FileCheck %s
# RUN: not llvm-mc -triple=aarch64 -filetype=obj %s --defsym ERR=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
# RUN: not llvm-mc -triple=aarch64 -filetype=obj %s --defsym OBJERR=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=OBJERR --implicit-check-not=error:

.globl g
g:
l:

# CHECK:      Section ({{.*}}) .rela.data {
# CHECK-NEXT:   0x0 R_AARCH64_PLT32 l 0x0
# CHECK-NEXT:   0x4 R_AARCH64_PLT32 l 0x4
# CHECK-NEXT:   0x8 R_AARCH64_PLT32 extern 0x4
# CHECK-NEXT:   0xC R_AARCH64_PLT32 g 0x8
# CHECK-NEXT:   0x10 R_AARCH64_PLT32 g 0x18
# CHECK-NEXT: }
.data
.word l@plt - .
.word l@plt - .data

.word extern@plt - . + 4
.word g@plt - . + 8
.word g@plt - .data + 8

# CHECK:      Section ({{.*}}) .rela.data1 {
# CHECK-NEXT:   0x0 R_AARCH64_GOTPCREL32 data1 0x0
# CHECK-NEXT:   0x4 R_AARCH64_GOTPCREL32 extern 0x4
# CHECK-NEXT:    0x8 R_AARCH64_GOTPCREL32 extern 0xFFFFFFFFFFFFFFFB
# CHECK-NEXT: }
.section .data1,"aw"
data1:
.word data1@GOTPCREL
.word extern@gotpcrel+4
.word extern@GOTPCREL-5

## Test parse-time errors
.ifdef ERR
# ERR: [[#@LINE+1]]:14: error: invalid variant 'pageoff'
.word extern@pageoff
.endif

## Test assemble-time errors
.ifdef OBJERR
# OBJERR: [[#@LINE+1]]:7: error: symbol 'und' can not be undefined in a subtraction expression
.word extern@plt - und

.quad g@plt - .

.word extern@gotpcrel - .

# OBJERR: [[#@LINE+1]]:7: error: symbol 'und' can not be undefined in a subtraction expression
.word extern@gotpcrel - und
.endif
