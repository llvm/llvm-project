# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t -debug-only=mc-dump-pre,mc-dump 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

#CHECK-LABEL:assembler backend - pre-layout
#      CHECK:MCSection Name:.text
#CHECK-LABEL:assembler backend - final-layout
#      CHECK:Sections:[
# CHECK-NEXT:MCSection Name:.text
# CHECK-NEXT:0 Data Size:0 []
# CHECK-NEXT:  Symbol @0 .text
# CHECK-NEXT:0 Align Align:4 Fill:0 FillLen:1 MaxBytesToEmit:4 Nops
# CHECK-NEXT:0 Data Size:0 []
# CHECK-NEXT:  Symbol @0 _start
# CHECK-NEXT:0 Org Offset:3 Value:0
# CHECK-NEXT:3 Relaxable Size:2 <MCInst #2001 <MCOperand Expr:.Ltmp0>>
# CHECK-NEXT:  Fixup @1 Value:.Ltmp0 Kind:4001
# CHECK-NEXT:5 Data Size:16 [48,8b,04,25,00,00,00,00,48,8b,04,25,00,00,00,00]
# CHECK-NEXT:  Fixup @4 Value:f0@<variant 11> Kind:4017
# CHECK-NEXT:  Fixup @12 Value:_start@<variant 11> Kind:4017
# CHECK-NEXT:  Symbol @16 .Ltmp0 Temporary
# CHECK-NEXT:MCSection Name:.data
# CHECK-NEXT:0 Data Size:0 []
# CHECK-NEXT:  Symbol @0 .data
# CHECK-NEXT:0 Align Align:4 Fill:0 FillLen:1 MaxBytesToEmit:4
# CHECK-NEXT:0 Data Size:4 [01,00,00,00]
# CHECK-NEXT:4 Fill Value:0 ValueSize:1 NumValues:1
# CHECK-NEXT:5 LEB Value:.Ltmp0-_start Signed:0
# CHECK-NEXT:]

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t -debug-only=mc-dump -save-temp-labels -g 2>&1 | FileCheck %s --check-prefix=CHECK2

#      CHECK2:5 Data Size:16 [48,8b,04,25,00,00,00,00,48,8b,04,25,00,00,00,00]
# CHECK2-NEXT:  Fixup @4 Value:f0@<variant 11> Kind:4017
# CHECK2-NEXT:  Fixup @12 Value:_start@<variant 11> Kind:4017
# CHECK2-NEXT:  Symbol @16 .Ltmp1
# CHECK2-NEXT:  Symbol @0 .Ltmp3 Temporary
# CHECK2-NEXT:  Symbol @8 .Ltmp4 Temporary
# CHECK2-NEXT:  Symbol @16 .Ltmp5 Temporary
# CHECK2-NEXT:  Symbol @16 .Lsec_end0 Temporary

_start:
var = _start
.org 3
jmp 1f
movq f0@GOTPCREL, %rax
movq _start@GOTPCREL, %rax
1:

.data
.p2align 2
.long 1
.space 1
.uleb128 1b-_start
