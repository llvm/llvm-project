; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Make sure pipeliner handle physical registers (e.g., used in
; inline asm

@g0 = external global ptr, align 4

; Function Attrs: nounwind
define i32 @f0(i32 %a0, ptr nocapture %a1) #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br label %b3

b3:                                               ; preds = %b3, %b2
  br i1 undef, label %b4, label %b3

b4:                                               ; preds = %b3
  br label %b5

b5:                                               ; preds = %b5, %b4
  %v0 = phi ptr [ inttoptr (i32 33554432 to ptr), %b4 ], [ %v4, %b5 ]
  %v1 = phi i32 [ 0, %b4 ], [ %v5, %b5 ]
  %v2 = ptrtoint ptr %v0 to i32
  tail call void asm sideeffect "    r1 = $1\0A    r0 = $0\0A    memw(r0) = r1\0A    dcfetch(r0)\0A", "r,r,~{r0},~{r1}"(i32 %v2, i32 %v1) #0
  %v3 = load ptr, ptr @g0, align 4
  %v4 = getelementptr inbounds i32, ptr %v3, i32 1
  store ptr %v4, ptr @g0, align 4
  %v5 = add nsw i32 %v1, 1
  %v6 = icmp eq i32 %v5, 200
  br i1 %v6, label %b6, label %b5

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b7, %b6
  br i1 undef, label %b8, label %b7

b8:                                               ; preds = %b7
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
