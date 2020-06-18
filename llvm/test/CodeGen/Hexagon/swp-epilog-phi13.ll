; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 -pipeliner-experimental-cg=true < %s | FileCheck %s

; Test epilogue generation when reading loop-carried dependency in stage 1 from
; stage 0. Make sure the illegal phi the expender creates gets cleaned up
; correctly during peeling

; CHECK: loop0
; CHECK: [[REG0:r([0-9]+)]] = add(r{{[0-9]+}},#8)
; CHECK: memw([[REG0]]+#0)
; CHECK: endloop0

; Function Attrs: nounwind
define i32* @f0(i16* nocapture readonly %a0, i32 %a1, i32 %a2, i32 %a3,  i16* %b) #0 {
b0:
  br i1 undef, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i16* [ %a0, %b1 ], [ %v2, %b2 ]
  %v2 = phi i16* [ undef, %b1 ], [ %v15, %b2 ]
  %v3 = phi i32* [ null, %b1 ], [ %v4, %b2 ]
  %v4 = phi i32* [ null, %b1 ], [ %v14, %b2 ]
  %v5 = phi i32 [ 0, %b1 ], [ %v13, %b2 ]
  %v6 = phi i16* [ undef, %b1 ], [ %v12, %b2 ]
  %a = mul i32 %v5, %a2
  %add = getelementptr inbounds i16, i16* %b, i32 %a
  %v7 = load i16, i16* %add, align 2
  %v8 = sext i16 %v7 to i32
  %v9 = call i32 @llvm.hexagon.M2.mpy.ll.s0(i32 %v8, i32 %v8) #2
  %v92 = call i32 @llvm.hexagon.M2.mpy.ll.s0(i32 %v9, i32 %v9) #2
  %v93 = call i32 @llvm.hexagon.M2.mpy.ll.s0(i32 %v92, i32 %v92) #2
  %v11 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32 %v8, i32 undef, i32 undef) #2
  store i32 %v11, i32* %v4, align 4
  %v12 = getelementptr inbounds i16, i16* %v6, i32 -1
  %v13 = add i32 %v5, 1
  %v14 = getelementptr inbounds i32, i32* %v3, i32 2
  store i32 %v93, i32* %v14, align 4
  %v15 = getelementptr inbounds i16, i16* %v1, i32 2
  %v16 = icmp slt i32 %v13, %a1
  br i1 %v16, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  %out = phi i32* [ null, %b0 ], [ %v14, %b2 ]
  ret i32* %out
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.ll.s0(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
