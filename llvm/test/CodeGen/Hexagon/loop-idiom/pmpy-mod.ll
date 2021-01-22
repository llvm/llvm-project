; Run -O2 to make sure that all the usual optimizations do happen before
; the Hexagon loop idiom recognition runs. This is to check that we still
; get this opportunity regardless of what happens before.

; RUN: opt -O2 -march=hexagon -S < %s | FileCheck %s
; RUN: opt -passes='default<O2>' -march=hexagon -S < %s | FileCheck %s

target triple = "hexagon"
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"

; CHECK-LABEL: define zeroext i16 @pmpy_mod_lsr
; There need to be two pmpy instructions.
; CHECK: call i64 @llvm.hexagon.M4.pmpyw
; CHECK: call i64 @llvm.hexagon.M4.pmpyw

define zeroext i16 @pmpy_mod_lsr(i8 zeroext %a0, i16 zeroext %a1) #0 {
b2:
  br label %b3

b3:                                               ; preds = %b44, %b2
  %v4 = phi i8 [ %a0, %b2 ], [ %v19, %b44 ]
  %v5 = phi i16 [ %a1, %b2 ], [ %v43, %b44 ]
  %v6 = phi i8 [ 0, %b2 ], [ %v45, %b44 ]
  %v7 = zext i8 %v6 to i32
  %v8 = icmp slt i32 %v7, 8
  br i1 %v8, label %b9, label %b46

b9:                                               ; preds = %b3
  %v10 = zext i8 %v4 to i32
  %v11 = and i32 %v10, 1
  %v12 = trunc i16 %v5 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = and i32 %v13, 1
  %v15 = xor i32 %v11, %v14
  %v16 = trunc i32 %v15 to i8
  %v17 = zext i8 %v4 to i32
  %v18 = ashr i32 %v17, 1
  %v19 = trunc i32 %v18 to i8
  %v20 = zext i8 %v16 to i32
  %v21 = icmp eq i32 %v20, 1
  br i1 %v21, label %b22, label %b26

b22:                                              ; preds = %b9
  %v23 = zext i16 %v5 to i32
  %v24 = xor i32 %v23, 16386
  %v25 = trunc i32 %v24 to i16
  br label %b27

b26:                                              ; preds = %b9
  br label %b27

b27:                                              ; preds = %b26, %b22
  %v28 = phi i16 [ %v25, %b22 ], [ %v5, %b26 ]
  %v29 = phi i8 [ 1, %b22 ], [ 0, %b26 ]
  %v30 = zext i16 %v28 to i32
  %v31 = ashr i32 %v30, 1
  %v32 = trunc i32 %v31 to i16
  %v33 = icmp ne i8 %v29, 0
  br i1 %v33, label %b34, label %b38

b34:                                              ; preds = %b27
  %v35 = zext i16 %v32 to i32
  %v36 = or i32 %v35, 32768
  %v37 = trunc i32 %v36 to i16
  br label %b42

b38:                                              ; preds = %b27
  %v39 = zext i16 %v32 to i32
  %v40 = and i32 %v39, 32767
  %v41 = trunc i32 %v40 to i16
  br label %b42

b42:                                              ; preds = %b38, %b34
  %v43 = phi i16 [ %v37, %b34 ], [ %v41, %b38 ]
  br label %b44

b44:                                              ; preds = %b42
  %v45 = add i8 %v6, 1
  br label %b3

b46:                                              ; preds = %b3
  ret i16 %v5
}

attributes #0 = { noinline nounwind "target-cpu"="hexagonv5" "target-features"="-hvx,-long-calls" }
