; RUN: opt < %s -passes=tsan -S | FileCheck %s
; Check that vptr reads are treated in a special way.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i8 @Foo(i8* %a) nounwind uwtable sanitize_thread {
entry:
; CHECK: call void @__tsan_vptr_read
  %0 = load i8, i8* %a, align 8, !tbaa !0
  ret i8 %0
}
!0 = !{!2, !2, i64 0}
!1 = !{!"Simple C/C++ TBAA"}
!2 = !{!"vtable pointer", !1}
