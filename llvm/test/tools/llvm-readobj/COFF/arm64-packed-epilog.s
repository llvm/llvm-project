// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck %s

// CHECK:          ExceptionData {
// CHECK-NEXT:       FunctionLength: 4
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       ExceptionData: Yes
// CHECK-NEXT:       EpiloguePacked: Yes
// CHECK-NEXT:       EpilogueOffset: 0
// CHECK-NEXT:       ByteCodeLength: 4
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       ExceptionHandler [
// CHECK-NEXT:         Routine: 0x11223344
// CHECK-NEXT:         Parameter: 0x55667788
// CHECK-NEXT:       ]

.section .pdata,"dr"
        .long func@IMGREL
        .long "$unwind$func"@IMGREL

        .text
        .globl  func
func:
        ret

.section .xdata,"dr"
"$unwind$func":
.byte 0x01, 0x00, 0x30, 0x08
.byte 0xe4, 0xe3, 0xe3, 0xe3
.byte 0x44, 0x33, 0x22, 0x11
.byte 0x88, 0x77, 0x66, 0x55
