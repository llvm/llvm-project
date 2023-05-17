; RUN: llc -march=hexagon -verify-machineinstrs < %s
; REQUIRES: asserts

; Test that the aggressive anti-dependence breaker does not attempt
; to rename a tied operand.

@g0 = external global [4 x i64], align 8
@g1 = external global [6 x i64], align 8
@g2 = external unnamed_addr constant [45 x i8], align 1
@g3 = external unnamed_addr constant [26 x i8], align 1
@g4 = external unnamed_addr constant [29 x i8], align 1
@g5 = external unnamed_addr constant [29 x i8], align 1

; Function Attrs: norecurse nounwind readonly
declare i64 @f0() #0

; Function Attrs: nounwind
define void @f1() #1 {
b0:
  %v0 = load i64, ptr @g0, align 8
  %v1 = trunc i64 %v0 to i32
  %v2 = load i64, ptr @g1, align 8
  %v3 = load i64, ptr getelementptr inbounds ([6 x i64], ptr @g1, i32 0, i32 3), align 8
  %v4 = lshr i64 %v2, 32
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, 0
  %v7 = trunc i64 %v3 to i32
  %v8 = lshr i64 %v3, 32
  %v9 = add i32 %v6, %v7
  %v10 = trunc i64 %v8 to i32
  %v11 = add i32 %v9, %v10
  %v12 = add i32 %v11, 0
  %v13 = add i32 %v12, 0
  %v14 = tail call i64 @f0()
  %v15 = lshr i64 %v0, 16
  %v16 = trunc i64 %v15 to i32
  %v17 = and i32 %v16, 65535
  %v18 = add nuw nsw i32 %v17, 0
  %v19 = zext i32 %v18 to i64
  %v20 = add i32 %v16, %v1
  %v21 = and i32 %v20, 65535
  %v22 = zext i32 %v21 to i64
  tail call void (ptr, ...) @f2(ptr @g2, i32 %v13) #2
  tail call void (ptr, ...) @f2(ptr @g3, i64 %v14) #2
  tail call void (ptr, ...) @f2(ptr @g4, i64 %v19) #2
  tail call void (ptr, ...) @f2(ptr @g5, i64 %v22) #2
  ret void
}

; Function Attrs: nounwind
declare void @f2(ptr nocapture readonly, ...) #1

attributes #0 = { norecurse nounwind readonly "target-cpu"="hexagonv55" }
attributes #1 = { nounwind "target-cpu"="hexagonv55" }
attributes #2 = { nounwind }
