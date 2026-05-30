; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s

@constant = constant i32 1

define void @main() {
  store i32 2, ptr @constant
  ret void
}

; CHECK: Entering function: main
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   store i32 2, ptr @constant, align 4 at @main
; CHECK-NEXT: Immediate UB detected: Try to write to a constant memory object at address 0x8.
; CHECK-NEXT: error: Execution of function 'main' failed.
