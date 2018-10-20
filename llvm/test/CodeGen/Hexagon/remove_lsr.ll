; Test fix for PR-13709.
; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: f0
; CHECK-NOT: lsr(r{{[0-9]+}}:{{[0-9]+}}, #32)
; CHECK-NOT: lsr(r{{[0-9]+}}:{{[0-9]+}}, #32)

; Convert the sequence
; r17:16 = lsr(r11:10, #32)
; .. = r16
; into
; r17:16 = lsr(r11:10, #32)
; .. = r11
; This makes the lsr instruction dead and it gets removed subsequently
; by a dead code removal pass.


%s.0 = type { i64 }
%s.1 = type { i32 }

define void @f0(%s.0* nocapture %a0, %s.1* nocapture %a1, %s.1* nocapture %a2, i8* nocapture %a3, i8* nocapture %a4) #0 {
b0:
  %v0 = getelementptr %s.0, %s.0* %a0, i32 1
  %v1 = getelementptr %s.1, %s.1* %a2, i32 1
  %v2 = getelementptr %s.1, %s.1* %a1, i32 1
  %v3 = getelementptr i8, i8* %a4, i32 1
  %v4 = getelementptr i8, i8* %a3, i32 1
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v5 = phi i32 [ %v38, %b1 ], [ 2, %b0 ]
  %v6 = phi i8* [ %v37, %b1 ], [ %v4, %b0 ]
  %v7 = phi i8* [ %v36, %b1 ], [ %v3, %b0 ]
  %v8 = phi %s.1* [ %v35, %b1 ], [ %v2, %b0 ]
  %v9 = phi %s.1* [ %v34, %b1 ], [ %v1, %b0 ]
  %v10 = phi %s.0* [ %v33, %b1 ], [ %v0, %b0 ]
  %v11 = phi i8 [ undef, %b0 ], [ %v30, %b1 ]
  %v12 = phi i8 [ undef, %b0 ], [ %v29, %b1 ]
  %v13 = phi i64 [ undef, %b0 ], [ %v28, %b1 ]
  %v14 = bitcast %s.1* %v8 to i32*
  %v15 = bitcast %s.1* %v9 to i32*
  %v16 = bitcast %s.0* %v10 to i64*
  %v17 = tail call i64 @llvm.hexagon.A2.vsubhs(i64 0, i64 %v13)
  %v18 = sext i8 %v12 to i32
  %v19 = trunc i64 %v13 to i32
  %v20 = trunc i64 %v17 to i32
  %v21 = tail call i32 @llvm.hexagon.C2.mux(i32 %v18, i32 %v19, i32 %v20)
  store i32 %v21, i32* %v14, align 4
  %v22 = sext i8 %v11 to i32
  %v23 = lshr i64 %v13, 32
  %v24 = trunc i64 %v23 to i32
  %v25 = lshr i64 %v17, 32
  %v26 = trunc i64 %v25 to i32
  %v27 = tail call i32 @llvm.hexagon.C2.mux(i32 %v22, i32 %v24, i32 %v26)
  store i32 %v27, i32* %v15, align 4
  %v28 = load i64, i64* %v16, align 8
  %v29 = load i8, i8* %v6, align 1
  %v30 = load i8, i8* %v7, align 1
  %v31 = trunc i32 %v5 to i8
  %v32 = icmp eq i8 %v31, 32
  %v33 = getelementptr %s.0, %s.0* %v10, i32 1
  %v34 = getelementptr %s.1, %s.1* %v9, i32 1
  %v35 = getelementptr %s.1, %s.1* %v8, i32 1
  %v36 = getelementptr i8, i8* %v7, i32 1
  %v37 = getelementptr i8, i8* %v6, i32 1
  %v38 = add i32 %v5, 1
  br i1 %v32, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

declare i64 @llvm.hexagon.A2.vsubhs(i64, i64) #1
declare i32 @llvm.hexagon.C2.mux(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv5" }
