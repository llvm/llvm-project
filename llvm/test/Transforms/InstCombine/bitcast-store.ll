; RUN: opt -S -instcombine < %s | FileCheck %s

; Instcombine should preserve metadata and alignment while
; folding a bitcast into a store.

; CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ([5 x i8*]* @G, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 16, !tag !0

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct.A = type { i32 (...)** }

@G = external constant [5 x i8*]

define void @foo(%struct.A* %a) nounwind {
entry:
  %0 = bitcast %struct.A* %a to i8***
  store i8** getelementptr inbounds ([5 x i8*]* @G, i64 0, i64 2), i8*** %0, align 16, !tag !0
  ret void
}

!0 = metadata !{metadata !"hello"}
