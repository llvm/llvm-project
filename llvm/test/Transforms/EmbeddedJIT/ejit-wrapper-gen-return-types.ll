; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s

; Return-value forwarding coverage. In the single-function mixed scheme the
; jit_dispatch block makes an indirect call through the specialized function
; pointer and must forward its result. Non-void scalar returns (integer wider
; than i32, pointer, floating point) must produce a typed indirect call whose
; result is returned, while the spliced fallback body keeps its own return.

; --- i64 return ---
; CHECK-LABEL: define i64 @ret_i64(i32 %a)
; CHECK: jit_fallback:
; CHECK: ret i64 0
; CHECK: jit_dispatch:
; CHECK: %[[R0:.*]] = call i64 {{.*}}(i32 %a)
; CHECK: ret i64 %[[R0]]
define i64 @ret_i64(i32 %a) !ejit.metadata !0 {
entry:
  ret i64 0
}

; --- pointer return, zero-arg function ---
; CHECK-LABEL: define ptr @ret_ptr()
; CHECK: jit_dispatch:
; CHECK: %[[R1:.*]] = call ptr {{.*}}()
; CHECK: ret ptr %[[R1]]
define ptr @ret_ptr() !ejit.metadata !0 {
entry:
  ret ptr null
}

; --- double return ---
; CHECK-LABEL: define double @ret_double(i32 %a)
; CHECK: jit_dispatch:
; CHECK: %[[R2:.*]] = call double {{.*}}(i32 %a)
; CHECK: ret double %[[R2]]
define double @ret_double(i32 %a) !ejit.metadata !0 {
entry:
  ret double 0.000000e+00
}

!0 = distinct !{!{!"ejit_entry"}}
