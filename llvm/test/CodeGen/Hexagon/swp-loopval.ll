; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Check that we correctly rename instructions that use a Phi's loop value,
; and the Phi and loop value are defined after the instruction.

%s.0 = type { [4 x i8], i16, i16, i32, [8 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [4 x %s.1], [4 x i8], i32, i32, [4 x i8], [14 x %s.2] }
%s.1 = type { i32, i32 }
%s.2 = type { [4 x i8] }

; Function Attrs: nounwind
define void @f0(%s.0* nocapture %a0) #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br label %b8

b3:                                               ; preds = %b9
  unreachable

b4:                                               ; preds = %b9
  br i1 undef, label %b7, label %b5

b5:                                               ; preds = %b4
  br i1 undef, label %b6, label %b7

b6:                                               ; preds = %b6, %b5
  %v0 = phi i32 [ %v10, %b6 ], [ 0, %b5 ]
  %v1 = load i32, i32* undef, align 4
  %v2 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 29, i32 %v0
  %v3 = bitcast %s.2* %v2 to i32*
  %v4 = load i32, i32* %v3, align 4
  %v5 = and i32 %v1, 65535
  %v6 = and i32 %v4, -65536
  %v7 = or i32 %v6, %v5
  %v8 = and i32 %v7, -2031617
  %v9 = or i32 %v8, 0
  store i32 %v9, i32* %v3, align 4
  %v10 = add nsw i32 %v0, 1
  %v11 = icmp eq i32 %v10, undef
  br i1 %v11, label %b7, label %b6

b7:                                               ; preds = %b6, %b5, %b4
  ret void

b8:                                               ; preds = %b8, %b2
  br i1 undef, label %b9, label %b8

b9:                                               ; preds = %b8
  br i1 undef, label %b3, label %b4
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
