; RUN: opt < %s -passes=tsan -S | FileCheck %s
; Check that tsan does not instrument reads from constant globals.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@const_global = external constant i32
define i32 @read_from_const_global() nounwind uwtable sanitize_thread readnone {
entry:
  %0 = load i32, ptr @const_global, align 4
  ret i32 %0
}
; CHECK: define i32 @read_from_const_global
; CHECK-NOT: __tsan
; CHECK: ret i32

@non_const_global = global i32 0, align 4
define i32 @read_from_non_const_global() nounwind uwtable sanitize_thread readonly {
entry:
  %0 = load i32, ptr @non_const_global, align 4
  ret i32 %0
}

; CHECK:  define i32 @read_from_non_const_global
; CHECK: __tsan_read
; CHECK: ret i32

@const_global_array = external constant [10 x i32]
define i32 @read_from_const_global_array(i32 %idx) nounwind uwtable sanitize_thread readnone {
entry:
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds [10 x i32], ptr @const_global_array, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

; CHECK: define i32 @read_from_const_global_array
; CHECK-NOT: __tsan
; CHECK: ret i32

%struct.Foo = type { ptr }
define void @call_virtual_func(ptr %f) uwtable sanitize_thread {
entry:
  %vtable = load ptr, ptr %f, align 8, !tbaa !2
  %0 = load ptr, ptr %vtable, align 8
  call void %0(ptr %f)
  ret void
}

; CHECK: define void @call_virtual_func
; CHECK: __tsan_vptr_read
; CHECK: = load
; CHECK-NOT: __tsan_read
; CHECK: = load
; CHECK: ret void

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"vtable pointer", !0}
!2 = !{!1, !1, i64 0}
