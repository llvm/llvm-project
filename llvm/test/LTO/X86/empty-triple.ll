; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1  2>&1 | FileCheck --allow-empty %s

; The test is to check the triple is set to default one when it's empty.
; Otherwise, an error will be raised by llvm-lto.

; CHECK-NOT: llvm-lto: error: No available targets are compatible with triple ""
