; RUN: opt < %s -tsan -S | FileCheck %s
; Check that tsan does not instrument reads from constant globals.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@const_global = external constant i32
define i32 @read_from_const_global() nounwind uwtable readnone {
entry:
  %0 = load i32* @const_global, align 4
  ret i32 %0
}
; CHECK: define i32 @read_from_const_global
; CHECK-NOT: __tsan
; CHECK: ret i32

@non_const_global = global i32 0, align 4
define i32 @read_from_non_const_global() nounwind uwtable readonly {
entry:
  %0 = load i32* @non_const_global, align 4
  ret i32 %0
}

; CHECK:  define i32 @read_from_non_const_global
; CHECK: __tsan_read
; CHECK: ret i32

@const_global_array = external constant [10 x i32]
define i32 @read_from_const_global_array(i32 %idx) nounwind uwtable readnone {
entry:
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds [10 x i32]* @const_global_array, i64 0, i64 %idxprom
  %0 = load i32* %arrayidx, align 4
  ret i32 %0
}

; CHECK: define i32 @read_from_const_global_array
; CHECK-NOT: __tsan
; CHECK: ret i32

%struct.Foo = type { i32 (...)** }
define void @call_virtual_func(%struct.Foo* %f) uwtable {
entry:
  %0 = bitcast %struct.Foo* %f to void (%struct.Foo*)***
  %vtable = load void (%struct.Foo*)*** %0, align 8, !tbaa !3
  %1 = load void (%struct.Foo*)** %vtable, align 8
  call void %1(%struct.Foo* %f)
  ret void
}

; CHECK: define void @call_virtual_func
; CHECK: __tsan_read
; CHECK: = load
; CHECK-NOT: __tsan_read
; CHECK: = load
; CHECK: ret void

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"vtable pointer", metadata !2}

