; RUN: llc -mtriple="i386-pc-mingw32" < %s | FileCheck %s
; PR5851

%0 = type { ptr }

define internal x86_stdcallcc void @MyFunc() nounwind {
entry:
; CHECK: MyFunc@0:
; CHECK: retl
  ret void
}

; PR14410
define x86_stdcallcc i32 @"\01DoNotMangle"(i32 %a) {
; CHECK: DoNotMangle:
; CHECK: retl $4
entry:
  ret i32 %a
}

%struct.large_type = type { i64, i64, i64 }

define x86_stdcallcc void @ReturnLargeType(ptr noalias nocapture sret(%struct.large_type) align 8 %agg.result) {
; CHECK: ReturnLargeType@0:
; CHECK: retl
entry:
  store i64 123, ptr %agg.result, align 8
  %b = getelementptr inbounds %struct.large_type, ptr %agg.result, i32 0, i32 1
  store i64 456, ptr %b, align 8
  %c = getelementptr inbounds %struct.large_type, ptr %agg.result, i32 0, i32 2
  store i64 789, ptr %c, align 8
  ret void
}

@B = global %0 { ptr @MyFunc }, align 4
; CHECK: _B:
; CHECK: .long _MyFunc@0

