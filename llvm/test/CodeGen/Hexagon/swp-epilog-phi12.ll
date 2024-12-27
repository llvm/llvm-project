; RUN: llc -mtriple=hexagon -hexagon-initial-cfg-cleanup=0 -pipeliner-experimental-cg=true -disable-cgp-delete-phis < %s | FileCheck %s

; Test epilogue generation when reading loop-carried dependency from a previous
; stage. The first epilogue should read value from iteration N-1 of the kernel.

; CHECK: loop0
; CHECK: r{{[0-9]+}} = add([[REG0:r([0-9]+)]],#8)
; CHECK: [[REG0]] = [[REG1:r([0-9]+)]]
; CHECK: endloop0
; CHECK: = add([[REG1]],#8)

; Function Attrs: nounwind
define ptr @f0(ptr nocapture readonly %a0, i32 %a1) #0 {
b0:
  %v0 = alloca [129 x i32], align 8
  br i1 undef, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi ptr [ %a0, %b1 ], [ %v2, %b2 ]
  %v2 = phi ptr [ undef, %b1 ], [ %v15, %b2 ]
  %v3 = phi ptr [ null, %b1 ], [ %v4, %b2 ]
  %v4 = phi ptr [ null, %b1 ], [ %v14, %b2 ]
  %v5 = phi i32 [ 0, %b1 ], [ %v13, %b2 ]
  %v6 = phi ptr [ undef, %b1 ], [ %v12, %b2 ]
  %v7 = load i16, ptr %v2, align 2
  %v8 = sext i16 %v7 to i32
  %v9 = call i32 @llvm.hexagon.M2.mpy.ll.s0(i32 %v8, i32 %v8) #2
  %v10 = load i16, ptr %v6, align 2
  %v11 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32 %v9, i32 undef, i32 undef) #2
  store i32 %v11, ptr %v4, align 4
  %v12 = getelementptr inbounds i16, ptr %v6, i32 -1
  %v13 = add i32 %v5, 1
  %v14 = getelementptr inbounds i32, ptr %v3, i32 2
  %v15 = getelementptr inbounds i16, ptr %v1, i32 2
  %v16 = icmp slt i32 %v13, %a1
  br i1 %v16, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  %out = phi ptr [ null, %b0 ], [ %v14, %b2 ]
  ret ptr %out
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.ll.s0(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
