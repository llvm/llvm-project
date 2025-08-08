# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj --triple=riscv64 --mattr=+relax %s -debug-only=mc-dump -o /dev/null 2>&1 | FileCheck %s

#      CHECK:Sections:[
# CHECK-NEXT:MCSection Name:.text
# CHECK-NEXT:0 Align Size:0+0 []
# CHECK-NEXT:  Align:4 Fill:0 FillLen:1 MaxBytesToEmit:4 Nops
# CHECK-NEXT:  Symbol @0 .text
# CHECK-NEXT:0 Data LinkerRelaxable Size:8 [97,00,00,00,e7,80,00,00]
# CHECK-NEXT:  Fixup @0 Value:specifier(19,ext) Kind:4023
# CHECK-NEXT:  Symbol @0 $x
# CHECK-NEXT:8 Align LinkerRelaxable Size:0+6 []
# CHECK-NEXT:  Align:8 Fill:0 FillLen:1 MaxBytesToEmit:8 Nops
# CHECK-NEXT:  Fixup @0 Value:6 Kind:[[#]]
# CHECK-NEXT:14 Align LinkerRelaxable Size:4+6 [13,05,30,00]
# CHECK-NEXT:  Align:8 Fill:0 FillLen:1 MaxBytesToEmit:8 Nops
# CHECK-NEXT:  Fixup @4 Value:6 Kind:[[#]]
# CHECK-NEXT:]

call ext
.p2align 3
li x10, 3
.p2align 3
