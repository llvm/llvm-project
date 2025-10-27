# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t -debug-only=mc-dump-pre,mc-dump -stats 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

#CHECK-LABEL:assembler backend - pre-layout
#      CHECK:MCSection Name:.text
#CHECK-LABEL:assembler backend - final-layout
#      CHECK:Sections:[
# CHECK-NEXT:MCSection Name:.text
# CHECK-NEXT:0 Align Size:0+0 []
# CHECK-NEXT:  Align:4 Fill:0 FillLen:1 MaxBytesToEmit:4 Nops
# CHECK-NEXT:  Symbol @0 .text
# CHECK-NEXT:0 Data Size:0 []
# CHECK-NEXT:  Symbol @0 _start
# CHECK-NEXT:  Symbol @0  Temporary
# CHECK-NEXT:0 Org Offset:3 Value:0
# CHECK-NEXT:3 Relaxable Size:0+2 [eb,00] <MCInst #[[#]] <MCOperand Expr:.Ltmp0>>
# CHECK-NEXT:  Fixup @1 Value:.Ltmp0 Kind:4001
# CHECK-NEXT:5 Data Size:16 [48,8b,04,25,00,00,00,00,48,8b,04,25,00,00,00,00]
# CHECK-NEXT:  Fixup @4 Value:f0@<variant 11> Kind:4017
# CHECK-NEXT:  Fixup @12 Value:_start@<variant 11> Kind:4017
# CHECK-NEXT:  Symbol @16 .Ltmp0 Temporary
# CHECK-NEXT:  Symbol @0  Temporary
# CHECK-NEXT:  Symbol @16  Temporary
# CHECK-NEXT:MCSection Name:.data
# CHECK-NEXT:0 Align Size:0+0 []
# CHECK-NEXT:  Align:4 Fill:0 FillLen:1 MaxBytesToEmit:4
# CHECK-NEXT:  Symbol @0 .data
# CHECK-NEXT:0 Data Size:4 [01,00,00,00]
# CHECK-NEXT:4 Fill Value:0 ValueSize:1 NumValues:1
# CHECK-NEXT:5 LEB Size:0+1 [15] Value:.Ltmp0-_start Signed:0
#      CHECK:]

# CHECK:  2 assembler         - Number of fixup evaluations for relaxation
# CHECK:  8 assembler         - Number of fixups

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t -debug-only=mc-dump -save-temp-labels -g 2>&1 | FileCheck %s --check-prefix=CHECK2

#      CHECK2:5 Data Size:16 [48,8b,04,25,00,00,00,00,48,8b,04,25,00,00,00,00]
# CHECK2-NEXT:  Fixup @4 Value:f0@<variant 11> Kind:4017
# CHECK2-NEXT:  Fixup @12 Value:_start@<variant 11> Kind:4017
# CHECK2-NEXT:  Symbol @16 .Ltmp2
# CHECK2-NEXT:  Symbol @0 .Lcfi0 Temporary
#      CHECK2:MCSection Name:.eh_frame
#      CHECK2:24 DwarfCallFrame Size:17+1 [00,00,00,00,00,00,00,00,00,00,00,00,00,00,00,00,00,45] AddrDelta:.Lcfi0-.Ltmp1
# CHECK2-NEXT:  Fixup @0 Value:.Ltmp12-.Ltmp11-0 Kind:4003

_start:
var = _start
.cfi_startproc
.org 3
jmp 1f
.cfi_offset %rbp, -24
movq f0@GOTPCREL, %rax
movq _start@GOTPCREL, %rax
1:
.cfi_endproc

.data
.p2align 2
.long 1
.space 1
.uleb128 1b-_start
