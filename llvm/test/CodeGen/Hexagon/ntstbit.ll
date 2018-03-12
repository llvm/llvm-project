; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: !tstbit

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = shl i32 1, %a2
  %v1 = and i32 %v0, %a1
  %v2 = icmp eq i32 %v1, 0
  br i1 %v2, label %b2, label %b1

b1:                                               ; preds = %b0
  tail call void bitcast (void (...)* @f1 to void ()*)() #0
  br label %b3

b2:                                               ; preds = %b0
  %v3 = tail call i32 bitcast (i32 (...)* @f2 to i32 ()*)() #0
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v4 = add nsw i32 %a1, 2
  %v5 = tail call i32 bitcast (i32 (...)* @f3 to i32 (i32, i32)*)(i32 %a0, i32 %v4) #0
  ret i32 0
}

declare void @f1(...)

declare i32 @f2(...)

declare i32 @f3(...)

attributes #0 = { nounwind }
