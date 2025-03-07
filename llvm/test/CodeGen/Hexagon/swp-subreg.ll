; RUN: llc -mtriple=hexagon -enable-pipeliner -stats -o /dev/null < %s 2>&1 -pipeliner-experimental-cg=true | FileCheck %s --check-prefix=STATS
; REQUIRES: asserts

; We're unable to pipeline a loop with a subreg as an operand of a Phi.

; STATS-NOT: 1 pipeliner   - Number of loops software pipelined

; Function Attrs: nounwind readnone
define void @f0(i32 %a0) #0 {
b0:
  %v0 = sext i32 %a0 to i64
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ 805306368, %b0 ], [ %v12, %b1 ]
  %v2 = phi i32 [ 5, %b0 ], [ %v13, %b1 ]
  %v3 = sext i32 %v1 to i64
  %v4 = mul nsw i64 %v3, %v0
  %v5 = lshr i64 %v4, 32
  %v6 = trunc i64 %v5 to i32
  %v7 = sub nsw i32 536870912, %v6
  %v8 = sext i32 %v7 to i64
  %v9 = mul nsw i64 %v8, %v3
  %v10 = lshr i64 %v9, 32
  %v11 = shl nuw nsw i64 %v10, 4
  %v12 = trunc i64 %v11 to i32
  %v13 = add nsw i32 %v2, -1
  %v14 = icmp eq i32 %v13, 0
  br i1 %v14, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv55" }
