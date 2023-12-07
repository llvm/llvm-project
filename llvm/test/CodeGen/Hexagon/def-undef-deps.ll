; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; The register coalescer created (via rematerialization) a definition of
; a register (R0), which had "undef" flag set. This caused the def to be
; ignored in the dependence graph, which then lead to an invalid instruction
; move in the machine scheduler (and an assert).
; The undef flags are already being cleared in the register cleanup, but
; that happens after register allocation. The undef flags need to be cleared
; earlier to avoid this issue.

%0 = type <{ ptr, ptr, i16, i8, i8, i8 }>
%1 = type { %2, %5, [3 x %3] }
%2 = type { %3, %4, i16, i16 }
%3 = type { i32, i32, i8, i8 }
%4 = type { i32, i32, i32 }
%5 = type { i8, i8, i8, i8, i32, i32, i16, i16, i32, i8, i8, i8, i32, i32, i16, i16, i32 }
%6 = type { %7, i8, i16, i16, i8, i8, i8, i8, i8 }
%7 = type { i32, i32, i16, i16, i16, i8 }

@g0 = external constant %0, align 1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.up(i32, i32) #1

declare void @f0(ptr, i32, i32, i32, i32, i32)

define void @f1(i8 zeroext %a0, ptr nocapture %a1, i8 zeroext %a2, i8 zeroext %a3) #0 {
b0:
  %v0 = getelementptr inbounds %1, ptr %a1, i32 0, i32 1, i32 9
  %v1 = load i8, ptr %v0, align 1
  %v2 = zext i8 %v1 to i32
  %v3 = getelementptr inbounds %1, ptr %a1, i32 0, i32 2, i32 %v2
  %v4 = tail call ptr @f2(i32 undef, i8 zeroext 0)
  br i1 undef, label %b1, label %b5

b1:                                               ; preds = %b0
  %v5 = tail call i32 @llvm.hexagon.M2.mpy.up(i32 undef, i32 undef)
  %v6 = tail call i32 @llvm.hexagon.M2.mpy.up(i32 undef, i32 undef)
  %v7 = zext i32 %v5 to i64
  %v8 = zext i32 %v6 to i64
  %v9 = add nuw nsw i64 %v8, %v7
  %v10 = lshr i64 %v9, 5
  %v11 = trunc i64 %v10 to i32
  store i32 %v11, ptr undef, align 4
  br i1 undef, label %b3, label %b2

b2:                                               ; preds = %b1
  store i32 0, ptr %v3, align 4
  tail call void @f0(ptr @g0, i32 undef, i32 0, i32 undef, i32 undef, i32 undef)
  br label %b4

b3:                                               ; preds = %b1
  br label %b4

b4:                                               ; preds = %b3, %b2
  unreachable

b5:                                               ; preds = %b0
  br i1 undef, label %b6, label %b7

b6:                                               ; preds = %b5
  unreachable

b7:                                               ; preds = %b5
  unreachable
}

declare ptr @f2(i32, i8 zeroext)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
