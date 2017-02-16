; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/remangle_intrinsics_tbaa.ll -o %t2.bc
; RUN: llvm-link -disable-lazy-loading %t2.bc %t1.bc -S | FileCheck %s

; Verify that we correctly rename the intrinsic and don't crash
; CHECK: @llvm.masked.store.v4p0s_some_named_struct.0s.p0v4p0s_some_named_struct.0s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%some_named_struct = type { i8 }

define void @foo(%some_named_struct*) {
  call void @llvm.masked.store.v4p0some_named_struct.p0v4p0some_named_struct(<4 x %some_named_struct*> undef, <4 x %some_named_struct*>* undef, i32 8, <4 x i1> undef), !tbaa !5
  ret void
}

declare void @llvm.masked.store.v4p0some_named_struct.p0v4p0some_named_struct(<4 x %some_named_struct*>, <4 x %some_named_struct*>*, i32, <4 x i1>) #1

!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
