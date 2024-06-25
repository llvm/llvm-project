; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that when we extract a byte from the result of a mask from predicate
; that the results of the mask all fit in the same word.
; CHECK: [[PRED:p[0-9]+]] = vcmpb.gtu(r{{.*}},#0)
; CHECK: [[REG1:r[0-9]*:[0-9]*]] = mask([[PRED]])
; CHECK: [[REG2:r[0-9]*]] = vtrunehb([[REG1]])
; CHECK: {{r[0-9]*}} = extractu([[REG2]],#1,#8)

target triple = "hexagon"

%struct.pluto = type { [12 x %struct.pluto.0], [4 x %struct.pluto.0], [2 x %struct.pluto.0], [4 x %struct.pluto.0], [6 x %struct.pluto.0], [2 x [7 x %struct.pluto.0]], [4 x %struct.pluto.0], [3 x [4 x %struct.pluto.0]], [3 x %struct.pluto.0], [3 x %struct.pluto.0] }
%struct.pluto.0 = type { i8, i8 }

@global = internal unnamed_addr constant [3 x [4 x [2 x i8]]] [[4 x [2 x i8]] [[2 x i8] c"\FAV", [2 x i8] c"\EF_", [2 x i8] c"\FA=", [2 x i8] c"\09-"], [4 x [2 x i8]] [[2 x i8] c"\06E", [2 x i8] c"\F3Z", [2 x i8] c"\004", [2 x i8] c"\08+"], [4 x [2 x i8]] [[2 x i8] c"\FA]", [2 x i8] c"\F2X", [2 x i8] c"\FA,", [2 x i8] c"\047"]], align 8

; Function Attrs: nofree noinline norecurse nosync nounwind memory(write)
define dso_local void @eggs(ptr nocapture %arg, ptr nocapture readnone %arg1, i32 %arg2, i32 %arg3, i32 %arg4) local_unnamed_addr #0 {
bb:
  %icmp = icmp sgt i32 %arg3, 0
  %select = select i1 %icmp, i32 %arg3, i32 0
  br i1 false, label %bb33, label %bb5

bb5:                                              ; preds = %bb
  %insertelement = insertelement <4 x i32> poison, i32 %select, i32 0
  %shufflevector = shufflevector <4 x i32> %insertelement, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %bb6

bb6:                                              ; preds = %bb6, %bb5
  %phi = phi i32 [ 0, %bb5 ], [ %add29, %bb6 ]
  %insertelement7 = insertelement <4 x i32> poison, i32 %phi, i32 0
  %shufflevector8 = shufflevector <4 x i32> %insertelement7, <4 x i32> poison, <4 x i32> zeroinitializer
  %add = add <4 x i32> %shufflevector8, <i32 0, i32 1, i32 2, i32 3>
  %add9 = add i32 %phi, 0
  %getelementptr = getelementptr inbounds [3 x [4 x [2 x i8]]], ptr @global, i32 0, i32 %arg2, i32 %add9, i32 0
  %getelementptr10 = getelementptr inbounds i8, ptr %getelementptr, i32 0
  %bitcast = bitcast ptr %getelementptr10 to ptr
  %load = load <8 x i8>, ptr %bitcast, align 1
  %shufflevector11 = shufflevector <8 x i8> %load, <8 x i8> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %shufflevector12 = shufflevector <8 x i8> %load, <8 x i8> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %getelementptr13 = getelementptr [3 x [4 x [2 x i8]]], ptr @global, i32 0, i32 %arg2, i32 %add9, i32 1
  %sext = sext <4 x i8> %shufflevector11 to <4 x i32>
  %mul = mul nsw <4 x i32> %shufflevector, %sext
  %ashr = ashr <4 x i32> %mul, <i32 4, i32 4, i32 4, i32 4>
  %sext14 = sext <4 x i8> %shufflevector12 to <4 x i32>
  %add15 = add nsw <4 x i32> %ashr, %sext14
  %icmp16 = icmp sgt <4 x i32> %add15, <i32 1, i32 1, i32 1, i32 1>
  %select17 = select <4 x i1> %icmp16, <4 x i32> %add15, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %icmp18 = icmp slt <4 x i32> %select17, <i32 126, i32 126, i32 126, i32 126>
  %select19 = select <4 x i1> %icmp18, <4 x i32> %select17, <4 x i32> <i32 126, i32 126, i32 126, i32 126>
  %icmp20 = icmp sgt <4 x i32> %select19, <i32 63, i32 63, i32 63, i32 63>
  %trunc = trunc <4 x i32> %select19 to <4 x i8>
  %add21 = add nsw <4 x i8> %trunc, <i8 -64, i8 -64, i8 -64, i8 -64>
  %getelementptr22 = getelementptr inbounds %struct.pluto, ptr %arg, i32 0, i32 1, i32 %add9, i32 0
  %sub = sub nsw <4 x i8> <i8 63, i8 63, i8 63, i8 63>, %trunc
  %select23 = select <4 x i1> %icmp20, <4 x i8> %add21, <4 x i8> %sub
  %getelementptr24 = getelementptr inbounds %struct.pluto, ptr %arg, i32 0, i32 1, i32 %add9, i32 1
  %zext = zext <4 x i1> %icmp20 to <4 x i8>
  %getelementptr25 = getelementptr inbounds i8, ptr %getelementptr24, i32 -1
  %bitcast26 = bitcast ptr %getelementptr25 to ptr
  %shufflevector27 = shufflevector <4 x i8> %select23, <4 x i8> %zext, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %shufflevector28 = shufflevector <8 x i8> %shufflevector27, <8 x i8> poison, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i8> %shufflevector28, ptr %bitcast26, align 1
  %add29 = add nuw i32 %phi, 4
  %icmp30 = icmp eq i32 %add29, 4
  br i1 %icmp30, label %bb31, label %bb6

bb31:                                             ; preds = %bb6
  %icmp32 = icmp eq i32 4, 4
  br i1 %icmp32, label %bb61, label %bb33

bb33:                                             ; preds = %bb31, %bb
  %phi34 = phi i32 [ 4, %bb31 ], [ 0, %bb ]
  br label %bb35

bb35:                                             ; preds = %bb35, %bb33
  %phi36 = phi i32 [ %phi34, %bb33 ], [ %add58, %bb35 ]
  %getelementptr37 = getelementptr inbounds [3 x [4 x [2 x i8]]], ptr @global, i32 0, i32 %arg2, i32 %phi36, i32 0
  %load38 = load i8, ptr %getelementptr37, align 2
  %getelementptr39 = getelementptr [3 x [4 x [2 x i8]]], ptr @global, i32 0, i32 %arg2, i32 %phi36, i32 1
  %load40 = load i8, ptr %getelementptr39, align 1
  %sext41 = sext i8 %load38 to i32
  %mul42 = mul nsw i32 %select, %sext41
  %ashr43 = ashr i32 %mul42, 4
  %sext44 = sext i8 %load40 to i32
  %add45 = add nsw i32 %ashr43, %sext44
  %icmp46 = icmp sgt i32 %add45, 1
  %select47 = select i1 %icmp46, i32 %add45, i32 1
  %icmp48 = icmp slt i32 %select47, 126
  %select49 = select i1 %icmp48, i32 %select47, i32 126
  %icmp50 = icmp sgt i32 %select49, 63
  %trunc51 = trunc i32 %select49 to i8
  %add52 = add nsw i8 %trunc51, -64
  %getelementptr53 = getelementptr inbounds %struct.pluto, ptr %arg, i32 0, i32 1, i32 %phi36, i32 0
  %sub54 = sub nsw i8 63, %trunc51
  %select55 = select i1 %icmp50, i8 %add52, i8 %sub54
  store i8 %select55, ptr %getelementptr53, align 1
  %getelementptr56 = getelementptr inbounds %struct.pluto, ptr %arg, i32 0, i32 1, i32 %phi36, i32 1
  %zext57 = zext i1 %icmp50 to i8
  store i8 %zext57, ptr %getelementptr56, align 1
  %add58 = add nuw nsw i32 %phi36, 1
  %icmp59 = icmp eq i32 %add58, 4
  br i1 %icmp59, label %bb60, label %bb35

bb60:                                             ; preds = %bb35
  br label %bb61

bb61:                                             ; preds = %bb60, %bb31
  ret void
}

attributes #0 = { nofree noinline norecurse nosync nounwind memory(write) "target-cpu"="hexagonv73" "target-features"="+hvx-length64b,+hvxv73,+v73" }
