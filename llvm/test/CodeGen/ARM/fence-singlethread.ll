; RUN: llc -mtriple=thumbv7-linux-gnueabihf %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-apple-ios %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-linux-gnueabihf %s -filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=OBJ

; OBJ-NOT: dmb

define void @fence_singlethread() {
; CHECK-LABEL: fence_singlethread:
; CHECK-NOT: dmb
; CHECK: @ COMPILER BARRIER
; CHECK-NOT: dmb

  fence singlethread seq_cst
  ret void
}
