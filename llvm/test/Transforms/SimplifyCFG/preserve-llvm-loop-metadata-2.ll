; RUN: opt -passes='simplifycfg' -S < %s | FileCheck %s

; CHECK: br i1 %2, label %loop2, label %loop1
; CHECK-SAME: llvm.loop

define void @test(i32 %1 ) {
.critedge:
  br label %loop1 

loop1:                                              ; preds = %loop2, .critedge 
  %111 = icmp eq i32 %1, 0
  br i1 %111, label %112, label %156

112:                                              ; preds = %loop1 
  br label %loop2 

loop2:                                              ; preds = %149, %112
  %148 = phi i1 [ false, %149 ], [ true, %112 ]
  br i1 %148, label %149, label %loop1, !llvm.loop !32

149:                                              ; preds = %loop2 
  br label %loop2 

156:                                              ; preds = loop1 
   ret void
} 

!32 = distinct !{!32, !33, !34}
!33 = !{!"llvm.loop.unroll.enable"}
!34 = !{!"llvm.loop.unroll.full"}
