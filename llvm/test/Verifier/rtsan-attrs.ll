; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attributes 'sanitize_realtime and nosanitize_realtime' are incompatible!
; CHECK-NEXT: ptr @sanitize_nosanitize
define void @sanitize_nosanitize() #0 {
  ret void
}

attributes #0 = { sanitize_realtime nosanitize_realtime }
