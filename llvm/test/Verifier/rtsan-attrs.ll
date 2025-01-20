; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attributes 'sanitize_realtime and sanitize_realtime_blocking' are incompatible!
; CHECK-NEXT: ptr @sanitize_unsafe
define void @sanitize_unsafe() #0 {
  ret void
}

attributes #0 = { sanitize_realtime sanitize_realtime_blocking }
