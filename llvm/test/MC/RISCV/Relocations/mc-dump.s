# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj --triple=riscv64 --mattr=+relax %s -debug-only=mc-dump -o /dev/null 2>&1 | FileCheck %s

#      CHECK:Sections:[
# CHECK-NEXT:MCSection Name:.text
# CHECK-NEXT:0 Data Size:0 []
# CHECK-NEXT:  Symbol @0 .text
# CHECK-NEXT:0 Align Align:4 Fill:0 FillLen:1 MaxBytesToEmit:4 Nops
# CHECK-NEXT:0 Data LinkerRelaxable Size:8 [97,00,00,00,e7,80,00,00]
# CHECK-NEXT:  Fixup @0 Value:specifier(19,ext) Kind:4023
# CHECK-NEXT:  Symbol @0 $x
# CHECK-NEXT:8 Align Align:8 Fill:0 FillLen:1 MaxBytesToEmit:8 Nops
# CHECK-NEXT:12 Data Size:4 [13,05,30,00]
# CHECK-NEXT:16 Align Align:8 Fill:0 FillLen:1 MaxBytesToEmit:8 Nops
# CHECK-NEXT:]

call ext
.p2align 3
li x10, 3
.p2align 3
