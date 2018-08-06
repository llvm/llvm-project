; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; This test caused an assert because there was a use of an instruction
; that was scheduled at stage 0, but no phi were added in the epilog.

%s.0 = type <{ i8*, i8*, i16, i8, i8, i8 }>
%s.1 = type { [4 x i16], [4 x i16], [4 x i16], [4 x i16], i32, i32, i32, i8, [10 x i32], [10 x [3 x i32]], [4 x i64], i8 }
%s.2 = type { [3 x i16], [4 x i8], i32, [3 x %s.3], [3 x %s.3], [3 x %s.3], [3 x %s.3], [3 x %s.3], [3 x %s.3], [6 x %s.3], [6 x %s.3], [6 x %s.3], i8, [3 x [3 x i16]], [3 x [3 x i16]], [3 x i16], [3 x i16], [6 x i16], [2 x i32], [10 x i32], [2 x i32], [2 x i32], [2 x [3 x i32]], [2 x i32], [2 x [3 x i64]], [2 x [3 x [3 x i32]]], [2 x [3 x i32]] }
%s.3 = type { i8, i8, i8, i8 }

@g0 = external constant %s.0, align 1

define void @f0(i8 zeroext %a0, i32 %a1, i32 %a2, i8 zeroext %a3, %s.1* nocapture %a4, %s.2* %a5, i8 zeroext %a6) #0 {
b0:
  br i1 undef, label %b1, label %b7

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b1
  unreachable

b3:                                               ; preds = %b1
  %v0 = select i1 undef, i32 2, i32 4
  %v1 = load i8, i8* undef, align 1
  %v2 = zext i8 %v1 to i32
  %v3 = icmp uge i32 %v2, %v0
  br label %b4

b4:                                               ; preds = %b4, %b3
  br i1 undef, label %b4, label %b8

b5:                                               ; preds = %b10
  unreachable

b6:                                               ; preds = %b10
  call void @f1(%s.0* @g0, i32 undef, i32 %v21, i32 undef, i32 undef)
  unreachable

b7:                                               ; preds = %b0
  ret void

b8:                                               ; preds = %b8, %b4
  %v4 = phi i32 [ %v11, %b8 ], [ undef, %b4 ]
  %v5 = phi i32 [ %v12, %b8 ], [ 0, %b4 ]
  %v6 = xor i1 false, %v3
  %v7 = zext i1 %v6 to i32
  %v8 = shl nuw nsw i32 %v7, 1
  %v9 = shl i32 %v4, 2
  %v10 = or i32 0, %v9
  %v11 = or i32 %v10, %v8
  %v12 = add i32 %v5, 1
  %v13 = icmp ult i32 %v12, %v0
  br i1 %v13, label %b8, label %b9

b9:                                               ; preds = %b9, %b8
  %v14 = phi i32 [ %v21, %b9 ], [ %v11, %b8 ]
  %v15 = icmp ne i32 undef, 1
  %v16 = xor i1 %v15, %v3
  %v17 = zext i1 %v16 to i32
  %v18 = shl nuw nsw i32 %v17, 1
  %v19 = shl i32 %v14, 2
  %v20 = or i32 0, %v19
  %v21 = or i32 %v20, %v18
  br i1 undef, label %b9, label %b10

b10:                                              ; preds = %b9
  br i1 undef, label %b6, label %b5
}

declare void @f1(%s.0*, i32, i32, i32, i32)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
