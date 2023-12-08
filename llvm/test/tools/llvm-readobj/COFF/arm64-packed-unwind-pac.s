// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck %s

// CHECK:      UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 44
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 2
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #-32]!
// CHECK-NEXT:       pacibsp
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

        .text
        .globl func
func:
        ret

        .section .pdata,"dr"
        .long func@IMGREL
        .long 0x0140002d // FunctionLength=11 RegF=0 RegI=0 H=0 CR=2 FrameSize=2
