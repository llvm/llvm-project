; RUN: llc -march=hexagon < %s | FileCheck %s

; Predicated (old) taken jumps weren't supported prior to V60. The purpose
; of this  test is to make sure that these instructions are not
; generated for V55.

; CHECK-NOT: if ({{!?}}p{{[0-3]}}) jump:t

%s.0 = type { %s.0*, i8 }

define i32 @f0(%s.0** nocapture %a0, i32 %a1) #0 {
b0:
  %v0 = and i32 %a1, 63
  %v1 = icmp eq i32 %v0, %a1
  br i1 %v1, label %b1, label %b7

b1:                                               ; preds = %b0
  %v2 = tail call i8* @f1()
  br label %b2

b2:                                               ; preds = %b4, %b1
  %v3 = phi i32 [ %v9, %b4 ], [ 0, %b1 ]
  %v4 = phi i32 [ %v5, %b4 ], [ 0, %b1 ]
  %v5 = add i32 %v4, 1
  %v6 = icmp ult i32 %v5, 7
  br i1 %v6, label %b3, label %b5

b3:                                               ; preds = %b2
  %v7 = tail call %s.0* @f2(i8* undef, i8* %v2)
  %v8 = icmp eq %s.0* %v7, null
  br i1 %v8, label %b7, label %b4

b4:                                               ; preds = %b3
  %v9 = select i1 undef, i32 1, i32 %v3
  br label %b2

b5:                                               ; preds = %b2
  br i1 undef, label %b7, label %b6

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b6, %b5, %b3, %b0
  %v10 = phi i32 [ -1, %b0 ], [ 1, %b6 ], [ %v3, %b5 ], [ -1, %b3 ]
  ret i32 %v10
}

declare i8* @f1()

declare %s.0* @f2(i8*, i8*)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
