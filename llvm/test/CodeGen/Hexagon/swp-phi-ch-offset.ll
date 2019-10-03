; RUN: llc -march=hexagon -enable-pipeliner -pipeliner-max-stages=2 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that we generate the correct offsets after we removed unneeded
; chain dependences between Phis and generated a better pipeline.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: = memd([[REG0:(r[0-9]+)]]+#8)
; CHECK: memd([[REG0]]++#8) =
; CHECK: }{{[ \t]*}}:endloop0

@g0 = common global [400 x i8] zeroinitializer, align 8
@g1 = common global [400 x i8] zeroinitializer, align 8

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br label %b2

b1:                                               ; preds = %b2
  ret void

b2:                                               ; preds = %b2, %b0
  %v0 = phi i8* [ getelementptr inbounds ([400 x i8], [400 x i8]* @g0, i32 0, i32 0), %b0 ], [ %v23, %b2 ]
  %v1 = phi i8* [ getelementptr inbounds ([400 x i8], [400 x i8]* @g1, i32 0, i32 0), %b0 ], [ %v24, %b2 ]
  %v2 = phi i32 [ 0, %b0 ], [ %v21, %b2 ]
  %v3 = bitcast i8* %v0 to <8 x i8>*
  %v4 = load <8 x i8>, <8 x i8>* %v3, align 8
  %v5 = bitcast i8* %v1 to <8 x i8>*
  %v6 = load <8 x i8>, <8 x i8>* %v5, align 8
  %v7 = bitcast <8 x i8> %v4 to <2 x i32>
  %v8 = extractelement <2 x i32> %v7, i32 0
  %v9 = extractelement <2 x i32> %v7, i32 1
  %v10 = bitcast <8 x i8> %v6 to <2 x i32>
  %v11 = extractelement <2 x i32> %v10, i32 0
  %v12 = extractelement <2 x i32> %v10, i32 1
  %v13 = tail call i64 @llvm.hexagon.S2.vzxtbh(i32 %v11)
  %v14 = tail call i64 @llvm.hexagon.S2.vzxtbh(i32 %v12)
  %v15 = tail call i64 @llvm.hexagon.M5.vmacbsu(i64 %v13, i32 %v8, i32 117901063)
  %v16 = tail call i64 @llvm.hexagon.M5.vmacbsu(i64 %v14, i32 %v9, i32 117901063)
  %v17 = tail call i32 @llvm.hexagon.S2.vtrunehb(i64 %v15)
  %v18 = tail call i32 @llvm.hexagon.S2.vtrunehb(i64 %v16)
  %v19 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v18, i32 %v17)
  %v20 = bitcast i64 %v19 to <8 x i8>
  store <8 x i8> %v20, <8 x i8>* %v5, align 8
  %v21 = add nsw i32 %v2, 8
  %v22 = icmp slt i32 %v2, 392
  %v23 = getelementptr i8, i8* %v0, i32 8
  %v24 = getelementptr i8, i8* %v1, i32 8
  br i1 %v22, label %b2, label %b1
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.vzxtbh(i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M5.vmacbsu(i64, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.vtrunehb(i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
