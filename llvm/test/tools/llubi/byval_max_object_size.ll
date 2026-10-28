; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s

target datalayout = "e-p:16:16"

define void @callee(ptr byval([32768 x i8]) %p) {
  ret void
}

define void @main() {
  %buf = alloca [32767 x i8]
  call void @callee(ptr byval([32768 x i8]) %buf)
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT:   %buf = alloca [32767 x i8], align 1 => ptr 0x8 [buf]
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   call void @callee(ptr byval([32768 x i8]) %buf) at @main <stdin>:11
; CHECK-NEXT: Immediate UB detected: Byval argument with size that exceeds the maximum object size. Size: 32768
; CHECK-NEXT: error: Execution of function 'main' failed.
