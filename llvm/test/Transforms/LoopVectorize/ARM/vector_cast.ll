; RUN: opt -passes=loop-vectorize -S -mattr=+neon < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabi"

; This requires the loop vectorizer to create an interleaved access group
; for the stores to the struct. Here we need to perform a bitcast from a vector
; of pointers to a vector i32s.

%class.A = type { ptr, i32 }

; CHECK-LABEL: test0
define void @test0(ptr %StartPtr, ptr %APtr) {
entry:
  br label %for.body.i

for.body.i:
  %addr = phi ptr [ %StartPtr, %entry ], [ %incdec.ptr.i, %for.body.i ]
  store ptr null, ptr %addr, align 4, !tbaa !8
  %Length.i.i = getelementptr inbounds %class.A, ptr %addr, i32 0, i32 1
  store i32 0, ptr %Length.i.i, align 4, !tbaa !11
  %incdec.ptr.i = getelementptr inbounds %class.A, ptr %addr, i32 1
  %cmp.i = icmp eq ptr %incdec.ptr.i, %APtr
  br i1 %cmp.i, label %exit, label %for.body.i

exit:
  ret void
}

!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !5, i64 0}
!9 = !{!"some struct", !5, i64 0, !10, i64 4}
!10 = !{!"int", !6, i64 0}
!11 = !{!9, !10, i64 4}
