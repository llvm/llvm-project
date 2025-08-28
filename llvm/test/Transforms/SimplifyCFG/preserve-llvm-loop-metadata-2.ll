; RUN: opt -passes='simplifycfg' -S < %s | FileCheck %s

; CHECK: br i1 %4, label %3, label %1, 
; CHECK-SAME: llvm.loop

define void @test(i32 %1 ) {
.critedge:
  br label %107

107:                                              ; preds = %147, .critedge 
  %111 = icmp eq i32 %1, 0
  br i1 %111, label %112, label %156

112:                                              ; preds = %107
  br label %147

147:                                              ; preds = %149, %112
  %148 = phi i1 [ false, %149 ], [ true, %112 ]
  br i1 %148, label %149, label %107, !llvm.loop !32

149:                                              ; preds = %147
  br label %147

156:                                              ; preds = %107
   ret void
} 

!32 = distinct !{!32, !33, !34}
!33 = !{!"llvm.loop.unroll.enable"}
!34 = !{!"llvm.loop.unroll.full"}
