; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that we generate the correct offsets for loads in the prolog
; after removing dependences on a post-increment instructions of the
; base register.

; CHECK: memh([[REG0:(r[0-9]+)]]+#0)
; CHECK: memh([[REG0]]+#2)
; CHECK: loop0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sath(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.asrh(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.addsat(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32, i32) #1

define void @f0() #0 align 2 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i16* [ undef, %b1 ], [ %v14, %b2 ]
  %v1 = phi i32 [ 0, %b1 ], [ %v12, %b2 ]
  %v2 = load i16, i16* %v0, align 2
  %v3 = sext i16 %v2 to i32
  %v4 = call i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32 undef, i32 %v3)
  %v5 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v4, i32 undef)
  %v6 = call i32 @llvm.hexagon.A2.addsat(i32 %v5, i32 32768)
  %v7 = call i32 @llvm.hexagon.A2.asrh(i32 %v6)
  %v8 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v7, i32 undef)
  %v9 = call i32 @llvm.hexagon.A2.sath(i32 %v8)
  %v10 = trunc i32 %v9 to i16
  store i16 %v10, i16* null, align 2
  %v11 = trunc i32 %v7 to i16
  store i16 %v11, i16* %v0, align 2
  %v12 = add nsw i32 %v1, 1
  %v13 = icmp slt i32 %v12, undef
  %v14 = getelementptr i16, i16* %v0, i32 1
  br i1 %v13, label %b2, label %b3

b3:                                               ; preds = %b2
  unreachable

b4:                                               ; No predecessors!
  unreachable
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
