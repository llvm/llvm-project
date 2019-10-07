; RUN: llc -march=hexagon -enable-pipeliner-opt-size -hexagon-initial-cfg-cleanup=0 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that we generate the correct names for the phis in the kernel for the
; incoming values. In this case, the loop contains a phi and has another phi
; as its loop definition, and the two phis are scheduled in different stages.
;
;    vreg5 = phi(x, vreg4) is scheduled in stage 1, cycle 0
;    vreg4 = phi(y, z) is scheduled in stage 0, cycle 0

; CHECK-DAG: :[[REG0:[0-9]+]]{{.*}} = {{.*}},#17
; CHECK-DAG: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: r{{[0-9]+}} = sxth(r[[REG0]])
; CHECK: endloop0

; Function Attrs: nounwind optsize
define void @f0() #0 {
b0:
  %v0 = getelementptr [8 x i16], [8 x i16]* undef, i32 0, i32 7
  %v1 = bitcast i16* %v0 to [8 x i16]*
  br label %b2

b1:                                               ; preds = %b2
  unreachable

b2:                                               ; preds = %b2, %b0
  %v2 = phi i32 [ 7, %b0 ], [ %v11, %b2 ]
  %v3 = phi i16 [ 17, %b0 ], [ %v7, %b2 ]
  %v4 = phi i16 [ 18, %b0 ], [ %v3, %b2 ]
  %v5 = sext i16 %v4 to i32
  %v6 = getelementptr i16, i16* null, i32 -2
  %v7 = load i16, i16* %v6, align 2
  %v8 = sext i16 %v7 to i32
  %v9 = tail call i32 @llvm.hexagon.A2.subsat(i32 %v5, i32 %v8)
  %v10 = trunc i32 %v9 to i16
  store i16 %v10, i16* null, align 2
  %v11 = add nsw i32 %v2, -1
  %v12 = icmp sgt i32 %v11, 1
  br i1 %v12, label %b2, label %b1
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subsat(i32, i32) #1

attributes #0 = { nounwind optsize "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
