; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0(i32 %a0) #0 {
b0:
  %v0 = ashr i32 %a0, 1
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ %v17, %b1 ], [ undef, %b0 ]
  %v2 = phi i32 [ %v19, %b1 ], [ 0, %b0 ]
  %v3 = phi i32 [ %v4, %b1 ], [ undef, %b0 ]
  %v4 = phi i32 [ %v14, %b1 ], [ undef, %b0 ]
  %v5 = phi i32 [ %v18, %b1 ], [ undef, %b0 ]
  %v6 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v1, i32 undef)
  %v7 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v3, i32 %v3)
  %v8 = tail call i64 @llvm.hexagon.S2.valignib(i64 %v6, i64 undef, i32 2)
  %v9 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 undef, i64 %v7, i64 %v8)
  %v10 = inttoptr i32 %v5 to i16*
  %v11 = load i16, i16* %v10, align 2
  %v12 = sext i16 %v11 to i32
  %v13 = add nsw i32 %v5, -8
  %v14 = tail call i32 @llvm.hexagon.A2.combine.ll(i32 %v12, i32 %v1)
  %v15 = inttoptr i32 %v13 to i16*
  %v16 = load i16, i16* %v15, align 2
  %v17 = sext i16 %v16 to i32
  %v18 = add nsw i32 %v5, -16
  %v19 = add nsw i32 %v2, 1
  %v20 = icmp eq i32 %v19, %v0
  br i1 %v20, label %b2, label %b1

b2:                                               ; preds = %b1
  %v21 = phi i64 [ %v9, %b1 ]
  %v22 = trunc i64 %v21 to i32
  %v23 = bitcast i8* undef to i32*
  store i32 %v22, i32* %v23, align 4
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.combine.ll(i32, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.valignib(i64, i64, i32) #1

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
attributes #2 = { noreturn nounwind }
