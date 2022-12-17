; RUN: opt -S -scalable-vectorization=on -force-target-supports-scalable-vectors=true -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we can vectorize loops which contain lifetime markers.

define void @test(i32 *%d) {
; CHECK-LABEL: @test(
; CHECK:      entry:
; CHECK:        [[ALLOCA:%.*]] = alloca [1024 x i32], align 16
; CHECK-NEXT:   [[BC:%.*]] = bitcast [1024 x i32]* [[ALLOCA]] to i8*
; CHECK:      vector.body:
; CHECK:        call void @llvm.lifetime.end.p0i8(i64 4096, i8* [[BC]])
; CHECK:        store <vscale x 2 x i32>
; CHECK:        call void @llvm.lifetime.start.p0i8(i64 4096, i8* [[BC]])

entry:
  %arr = alloca [1024 x i32], align 16
  %0 = bitcast [1024 x i32]* %arr to i8*
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* %0) #1
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* %0) #1
  %arrayidx = getelementptr inbounds i32, i32* %d, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx, align 8
  store i32 100, i32* %arrayidx, align 8
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* %0) #1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end, !llvm.loop !0

for.end:
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* %0) #1
  ret void
}

; CHECK-LABEL: @testloopvariant(
; CHECK:      entry:
; CHECK:        [[ALLOCA:%.*]] = alloca [1024 x i32], align 16
; CHECK:      vector.ph:
; CHECK:        [[TMP1:%.*]] = insertelement <vscale x 2 x [1024 x i32]*> poison, [1024 x i32]* %arr, i32 0
; CHECK-NEXT:   [[SPLAT_ALLOCA:%.*]] = shufflevector <vscale x 2 x [1024 x i32]*> [[TMP1]], <vscale x 2 x [1024 x i32]*> poison, <vscale x 2 x i32> zeroinitializer
; CHECK:      vector.body:
; CHECK:        [[BC_ALLOCA:%.*]] = bitcast <vscale x 2 x [1024 x i32]*> [[SPLAT_ALLOCA]] to <vscale x 2 x i8*>
; CHECK-NEXT:   [[ONE_LIFETIME:%.*]] = extractelement <vscale x 2 x i8*> [[BC_ALLOCA]], i32 0
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 4096, i8* [[ONE_LIFETIME]])
; CHECK:        store <vscale x 2 x i32>
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 4096, i8* [[ONE_LIFETIME]])

define void @testloopvariant(i32 *%d) {
entry:
  %arr = alloca [1024 x i32], align 16
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr [1024 x i32], [1024 x i32]* %arr, i32 0, i64 %indvars.iv
  %1 = bitcast [1024 x i32]* %arr to i8*
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* %1) #1
  %arrayidx = getelementptr inbounds i32, i32* %d, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx, align 8
  store i32 100, i32* %arrayidx, align 8
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* %1) #1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end, !llvm.loop !0

for.end:
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
