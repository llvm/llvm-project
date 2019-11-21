; RUN: llc -mtriple aarch64-arm-linux-gnu -o - %s | FileCheck %s

; CHECK:        a:                                      // @a
; CHECK-NEXT:   // %bb.0:
; CHECK-NEXT:           sub     sp, sp, #16
; CHECK-NOT:            .cfi{{.*}}
; CHECK:                ret
define void @a() nounwind {
  %1 = alloca i32, align 4
  store i32 1, i32* %1, align 4
  ret void
}

