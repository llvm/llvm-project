# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t -debug-only=mc-dump 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

#CHECK-LABEL:assembler backend - pre-layout
#      CHECK:MCSection Name:.text
#CHECK-LABEL:assembler backend - final-layout
#      CHECK:Sections:[
# CHECK-NEXT:MCSection Name:.text
# CHECK-NEXT:0 Data Size:0 []
# CHECK-NEXT:0 Align Align:4 Value:0 ValueSize:1 MaxBytesToEmit:4 Nops
# CHECK-NEXT:0 Data Size:0 []
# CHECK-NEXT:0 Org Offset:3 Value:0
# CHECK-NEXT:3 Relaxable Size:2 <MCInst #1996 <MCOperand Expr:.Ltmp1>>
# CHECK-NEXT:  Fixup Offset:1 Value:.Ltmp1-1 Kind:4006
# CHECK-NEXT:5 Data Size:16 [48,8b,04,25,00,00,00,00,48,8b,04,25,00,00,00,00]
# CHECK-NEXT:  Fixup Offset:4 Value:f0@<variant 11> Kind:4021
# CHECK-NEXT:  Fixup Offset:12 Value:f1@<variant 11> Kind:4021
# CHECK-NEXT:MCSection Name:.data
# CHECK-NEXT:0 Data Size:0 []
# CHECK-NEXT:0 Align Align:4 Value:0 ValueSize:1 MaxBytesToEmit:4
# CHECK-NEXT:0 Data Size:4 [01,00,00,00]
# CHECK-NEXT:4 Fill Value:0 ValueSize:1 NumValues:1
# CHECK-NEXT:5 LEB Value:.Ltmp1-.Ltmp0 Signed:0
# CHECK-NEXT:]
# CHECK-NEXT:Symbols:[
# CHECK-NEXT:(.text, Index:0, )
# CHECK-NEXT:(.Ltmp0, Index:0, )

0:
.org 3
jmp 1f
movq f0@GOTPCREL, %rax
movq f1@GOTPCREL, %rax
1:

.data
.p2align 2
.long 1
.space 1
.uleb128 1b-0b
