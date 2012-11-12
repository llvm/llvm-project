; RUN: opt < %s -tsan -S | FileCheck %s
; Check that vtable pointer updates are treated in a special way.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @Foo(i8** nocapture %a, i8* %b) nounwind uwtable {
entry:
; CHECK: call void @__tsan_vptr_update
  store i8* %b, i8** %a, align 8, !tbaa !0
  ret void
}
!0 = metadata !{metadata !"vtable pointer", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA", null}

