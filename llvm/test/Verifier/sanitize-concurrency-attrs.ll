; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attributes 'sanitize_thread and sanitize_concurrency' are incompatible!
; CHECK-NEXT: ptr @sanitize_unsafe
define void @sanitize_unsafe() #0 {
  ret void
}

attributes #0 = { sanitize_thread sanitize_concurrency }
