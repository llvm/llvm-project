; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

define i32 @test_switch(i32 %i, i32 %a, i32 %b) #0 {
; CHECK: define i32 @test_switch(i32 [[I:%.*]], i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK: [[ENTRY:.*]]:
; CHECK-NEXT: switch i32 [[I]], label %[[DEFAULT:.*]] [
; CHECK-NEXT:   i32 0, label %[[RETURN:.*]]
; CHECK-NEXT:   i32 1, label %[[BB1:.*]]
; CHECK-NEXT: ]
; 
; CHECK: [[BB1]]:
; CHECK-NEXT: br label %[[RETURN]]
; 
; CHECK: [[DEFAULT]]:
; CHECK-NEXT: br label %[[RETURN]]
; 
; CHECK: [[RETURN]]
; CHECK-NEXT: [[RETVAL:%.*]] = phi i32 [ -1, %[[DEFAULT]] ], [ [[B]], %[[BB1]] ], [ [[A]], %[[ENTRY]] ]
; CHECK-NEXT: ret i32 [[RETVAL]]
; 
entry:
  switch i32 %i, label %sw.default [
    i32 0, label %return
    i32 1, label %sw.bb1
  ]

sw.bb1:                                           ; preds = %entry
  br label %return

sw.default:                                       ; preds = %entry
  br label %return

return:                                           ; preds = %entry, %sw.default, %sw.bb1
  %retval.0 = phi i32 [ -1, %sw.default ], [ %b, %sw.bb1 ], [ %a, %entry ]
  ret i32 %retval.0
}

attributes #0 = { nounwind memory(none) }
