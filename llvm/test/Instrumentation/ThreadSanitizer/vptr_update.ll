; RUN: opt < %s -passes=tsan -S | FileCheck %s
; Check that vtable pointer updates are treated in a special way.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @Foo(ptr nocapture %a, ptr %b) nounwind uwtable sanitize_thread {
entry:
; CHECK-LABEL: @Foo
; CHECK: call void @__tsan_vptr_update
; CHECK: ret void
  store ptr %b, ptr %a, align 8, !tbaa !0
  ret void
}

define void @FooInt(ptr nocapture %a, i64 %b) nounwind uwtable sanitize_thread {
entry:
; CHECK-LABEL: @FooInt
; CHECK: call void @__tsan_vptr_update
; CHECK: ret void
  store i64 %b, ptr %a, align 8, !tbaa !0
  ret void
}


declare i32 @Func1()
declare i32 @Func2()

; Test that we properly handle vector stores marked as vtable updates.
define void @VectorVptrUpdate(ptr nocapture %a, ptr %b) nounwind uwtable sanitize_thread {
entry:
; CHECK-LABEL: @VectorVptrUpdate
; CHECK: call void @__tsan_vptr_update{{.*}}Func1
; CHECK-NOT: call void @__tsan_vptr_update
; CHECK: ret void
  store <2 x ptr> <ptr @Func1, ptr @Func2>,  ptr %a, align 8, !tbaa !0
  ret void
}

!0 = !{!2, !2, i64 0}
!1 = !{!"Simple C/C++ TBAA"}
!2 = !{!"vtable pointer", !1}
