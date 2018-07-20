; RUN: llc -march=hexagon -enable-pipeliner -pipeliner-max-stages=1 < %s | FileCheck %s

; Test that we update the offset correctly for loads that are
; moved past stores. In these cases, we change the dependences
; to make it easier to move the instructions, and we have to update
; the register/offsets correctly after the schedule is finalized.

@g0 = common global [400 x i32] zeroinitializer, align 8
@g1 = common global [400 x i32] zeroinitializer, align 8

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br label %b2

b1:                                               ; preds = %b2
  ret void

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: = memd([[REG1:(r[0-9]+)]]+#8)
; CHECK: memd([[REG1]]++#8) =
; CHECK: }{{[ \t]*}}:endloop

b2:                                               ; preds = %b2, %b0
  %v0 = phi i32* [ getelementptr inbounds ([400 x i32], [400 x i32]* @g0, i32 0, i32 0), %b0 ], [ %v11, %b2 ]
  %v1 = phi i32* [ getelementptr inbounds ([400 x i32], [400 x i32]* @g1, i32 0, i32 0), %b0 ], [ %v12, %b2 ]
  %v2 = phi i32 [ 0, %b0 ], [ %v9, %b2 ]
  %v3 = bitcast i32* %v0 to <2 x i32>*
  %v4 = load <2 x i32>, <2 x i32>* %v3, align 8
  %v5 = mul <2 x i32> %v4, <i32 7, i32 7>
  %v6 = bitcast i32* %v1 to <2 x i32>*
  %v7 = load <2 x i32>, <2 x i32>* %v6, align 8
  %v8 = add <2 x i32> %v7, %v5
  store <2 x i32> %v8, <2 x i32>* %v6, align 8
  %v9 = add nsw i32 %v2, 2
  %v10 = icmp slt i32 %v2, 398
  %v11 = getelementptr i32, i32* %v0, i32 2
  %v12 = getelementptr i32, i32* %v1, i32 2
  br i1 %v10, label %b2, label %b1
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
