; RUN: opt < %s -aarch64-stack-tagging -S -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use(i8*)
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)

define void @OneVarNoInit() sanitize_memtag {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0)
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @OneVarNoInit(
; CHECK-DAG:  [[X:%.*]] = alloca { i32, [12 x i8] }, align 16
; CHECK-DAG:  [[TX:%.*]] = call { i32, [12 x i8] }* @llvm.aarch64.tagp.{{.*}}({ i32, [12 x i8] }* [[X]], {{.*}}, i64 0)
; CHECK-DAG:  [[TX32:%.*]] = bitcast { i32, [12 x i8] }* [[TX]] to i32*
; CHECK-DAG:  [[TX8:%.*]] = bitcast i32* [[TX32]] to i8*
; CHECK-DAG:  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull [[TX8]])
; CHECK-DAG:  call void @llvm.aarch64.settag(i8* [[TX8]], i64 16)
; CHECK-DAG:  call void @use(i8* nonnull [[TX8]])
; CHECK-DAG:  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull [[TX8]])

define void @OneVarInitConst() sanitize_memtag {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0)
  store i32 42, i32* %x, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @OneVarInitConst(
; CHECK:  [[TX:%.*]] = call { i32, [12 x i8] }* @llvm.aarch64.tagp
; CHECK:  [[TX32:%.*]] = bitcast { i32, [12 x i8] }* [[TX]] to i32*
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX32]] to i8*
; CHECK-NOT: aarch64.settag
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8]], i64 42, i64 0)
; Untagging before lifetime.end:
; CHECK:  call void @llvm.aarch64.settag(
; CHECK-NOT: aarch64.settag
; CHECK:  ret void

define void @ArrayInitConst() sanitize_memtag {
entry:
  %x = alloca i32, i32 16, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %0)
  store i32 42, i32* %x, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @ArrayInitConst(
; CHECK:  [[TX:%.*]] = call i32* @llvm.aarch64.tagp.
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX]] to i8*
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8]], i64 42, i64 0)
; CHECK:  [[TX8_16:%.*]] = getelementptr i8, i8* [[TX8]], i32 16
; CHECK:  call void @llvm.aarch64.settag.zero(i8* [[TX8_16]], i64 48)
; CHECK:  ret void

define void @ArrayInitConst2() sanitize_memtag {
entry:
  %x = alloca i32, i32 16, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %0)
  store i32 42, i32* %x, align 4
  %1 = getelementptr i32, i32* %x, i32 1
  store i32 43, i32* %1, align 4
  %2 = getelementptr i32, i32* %x, i32 2
  %3 = bitcast i32* %2 to i64*
  store i64 -1, i64* %3, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @ArrayInitConst2(
; CHECK:  [[TX:%.*]] = call i32* @llvm.aarch64.tagp.
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX]] to i8*
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8]], i64 184683593770, i64 -1)
; CHECK:  [[TX8_16:%.*]] = getelementptr i8, i8* [[TX8]], i32 16
; CHECK:  call void @llvm.aarch64.settag.zero(i8* [[TX8_16]], i64 48)
; CHECK:  ret void

define void @ArrayInitConstSplit() sanitize_memtag {
entry:
  %x = alloca i32, i32 16, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %0)
  %1 = getelementptr i32, i32* %x, i32 1
  %2 = bitcast i32* %1 to i64*
  store i64 -1, i64* %2, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @ArrayInitConstSplit(
; CHECK:  [[TX:%.*]] = call i32* @llvm.aarch64.tagp.
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX]] to i8*
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8]], i64 -4294967296, i64 4294967295)
; CHECK:  ret void

define void @ArrayInitConstWithHoles() sanitize_memtag {
entry:
  %x = alloca i32, i32 32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 128, i8* nonnull %0)
  %1 = getelementptr i32, i32* %x, i32 5
  store i32 42, i32* %1, align 4
  %2 = getelementptr i32, i32* %x, i32 14
  store i32 43, i32* %2, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 128, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @ArrayInitConstWithHoles(
; CHECK:  [[TX:%.*]] = call i32* @llvm.aarch64.tagp.
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX]] to i8*
; CHECK:  call void @llvm.aarch64.settag.zero(i8* [[TX8]], i64 16)
; CHECK:  [[TX8_16:%.*]] = getelementptr i8, i8* %0, i32 16
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8_16]], i64 180388626432, i64 0)
; CHECK:  [[TX8_32:%.*]] = getelementptr i8, i8* %0, i32 32
; CHECK:  call void @llvm.aarch64.settag.zero(i8* [[TX8_32]], i64 16)
; CHECK:  [[TX8_48:%.*]] = getelementptr i8, i8* %0, i32 48
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8_48]], i64 0, i64 43)
; CHECK:  [[TX8_64:%.*]] = getelementptr i8, i8* %0, i32 64
; CHECK:  call void @llvm.aarch64.settag.zero(i8* [[TX8_64]], i64 64)
; CHECK:  ret void

define void @InitNonConst(i32 %v) sanitize_memtag {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0)
  store i32 %v, i32* %x, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @InitNonConst(
; CHECK:  [[TX:%.*]] = call { i32, [12 x i8] }* @llvm.aarch64.tagp
; CHECK:  [[TX32:%.*]] = bitcast { i32, [12 x i8] }* [[TX]] to i32*
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX32]] to i8*
; CHECK:  [[V:%.*]] = zext i32 %v to i64
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8]], i64 [[V]], i64 0)
; CHECK:  ret void

define void @InitNonConst2(i32 %v, i32 %w) sanitize_memtag {
entry:
  %x = alloca i32, i32 4, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  store i32 %v, i32* %x, align 4
  %1 = getelementptr i32, i32* %x, i32 1
  store i32 %w, i32* %1, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @InitNonConst2(
; CHECK:  [[TX:%.*]] = call i32* @llvm.aarch64.tagp
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX]] to i8*
; CHECK:  [[V:%.*]] = zext i32 %v to i64
; CHECK:  [[W:%.*]] = zext i32 %w to i64
; CHECK:  [[WS:%.*]] = shl i64 [[W]], 32
; CHECK:  [[VW:%.*]] = or i64 [[V]], [[WS]]
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8]], i64 [[VW]], i64 0)
; CHECK:  ret void

define void @InitVector() sanitize_memtag {
entry:
  %x = alloca i32, i32 4, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  %1 = bitcast i32* %x to <2 x i32>*
  store <2 x i32> <i32 1, i32 2>, <2 x i32>* %1, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @InitVector(
; CHECK:  [[TX:%.*]] = call i32* @llvm.aarch64.tagp
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX]] to i8*
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8]], i64 bitcast (<2 x i32> <i32 1, i32 2> to i64), i64 0)
; CHECK:  ret void

define void @InitVectorPtr(i32* %p) sanitize_memtag {
entry:
  %s = alloca <4 x i32*>, align 8
  %v0 = insertelement <4 x i32*> undef, i32* %p, i32 0
  %v1 = shufflevector <4 x i32*> %v0, <4 x i32*> undef, <4 x i32> zeroinitializer
  store <4 x i32*> %v1, <4 x i32*>* %s
  %0 = bitcast <4 x i32*>* %s to i8*
  call void @use(i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @InitVectorPtr(
; CHECK:  call <4 x i32*>* @llvm.aarch64.tagp
; CHECK:  [[V1:%.*]] = shufflevector
; CHECK:  [[V2:%.*]] = ptrtoint <4 x i32*> [[V1]] to <4 x i64>
; CHECK:  [[V3:%.*]] = bitcast <4 x i64> [[V2]] to i256
; CHECK:  [[A1:%.*]] = trunc i256 [[V3]] to i64
; CHECK:  [[A2_:%.*]] = lshr i256 [[V3]], 64
; CHECK:  [[A2:%.*]] = trunc i256 [[A2_]] to i64
; CHECK:  [[A3_:%.*]] = lshr i256 [[V3]], 128
; CHECK:  [[A3:%.*]] = trunc i256 [[A3_]] to i64
; CHECK:  [[A4_:%.*]] = lshr i256 [[V3]], 192
; CHECK:  [[A4:%.*]] = trunc i256 [[A4_]] to i64
; CHECK:  call void @llvm.aarch64.stgp({{.*}}, i64 [[A1]], i64 [[A2]])
; CHECK:  call void @llvm.aarch64.stgp({{.*}}, i64 [[A3]], i64 [[A4]])
; CHECK:  ret void

define void @InitVectorSplit() sanitize_memtag {
entry:
  %x = alloca i32, i32 4, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  %1 = getelementptr i32, i32* %x, i32 1
  %2 = bitcast i32* %1 to <2 x i32>*
  store <2 x i32> <i32 1, i32 2>, <2 x i32>* %2, align 4
  call void @use(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @InitVectorSplit(
; CHECK:  [[TX:%.*]] = call i32* @llvm.aarch64.tagp
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX]] to i8*
; CHECK:  call void @llvm.aarch64.stgp(i8* [[TX8]], i64 shl (i64 bitcast (<2 x i32> <i32 1, i32 2> to i64), i64 32), i64 lshr (i64 bitcast (<2 x i32> <i32 1, i32 2> to i64), i64 32))
; CHECK:  ret void

define void @MemSetZero() sanitize_memtag {
entry:
  %x = alloca i32, i32 8, align 16
  %0 = bitcast i32* %x to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %0, i8 0, i64 32, i1 false)
  call void @use(i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @MemSetZero(
; CHECK:  [[TX:%.*]] = call i32* @llvm.aarch64.tagp
; CHECK:  [[TX8:%.*]] = bitcast i32* [[TX]] to i8*
; CHECK:  call void @llvm.aarch64.settag.zero(i8* [[TX8]], i64 32)
; CHECK:  ret void


define void @MemSetNonZero() sanitize_memtag {
entry:
  %x = alloca i32, i32 8, align 16
  %0 = bitcast i32* %x to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %0, i8 42, i64 32, i1 false)
  call void @use(i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @MemSetNonZero(
; CHECK:  call void @llvm.aarch64.stgp(i8* {{.*}}, i64 3038287259199220266, i64 3038287259199220266)
; CHECK:  call void @llvm.aarch64.stgp(i8* {{.*}}, i64 3038287259199220266, i64 3038287259199220266)
; CHECK:  ret void


define void @MemSetNonZero2() sanitize_memtag {
entry:
  %x = alloca [32 x i8], align 16
  %0 = getelementptr inbounds [32 x i8], [32 x i8]* %x, i64 0, i64 2
  call void @llvm.memset.p0i8.i64(i8* nonnull %0, i8 42, i64 28, i1 false)
  call void @use(i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @MemSetNonZero2(
; CHECK:  call void @llvm.aarch64.stgp(i8* {{.*}}, i64 3038287259199209472, i64 3038287259199220266)
; CHECK:  call void @llvm.aarch64.stgp(i8* {{.*}}, i64 3038287259199220266, i64 46360584399402)
; CHECK:  ret void

define void @MemSetNonZero3() sanitize_memtag {
entry:
  %x = alloca [32 x i8], align 16
  %0 = getelementptr inbounds [32 x i8], [32 x i8]* %x, i64 0, i64 2
  call void @llvm.memset.p0i8.i64(i8* nonnull %0, i8 42, i64 4, i1 false)
  %1 = getelementptr inbounds [32 x i8], [32 x i8]* %x, i64 0, i64 24
  call void @llvm.memset.p0i8.i64(i8* nonnull %1, i8 42, i64 8, i1 false)
  call void @use(i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @MemSetNonZero3(
; CHECK:  call void @llvm.aarch64.stgp(i8* {{.*}}, i64 46360584388608, i64 0)
; CHECK:  call void @llvm.aarch64.stgp(i8* {{.*}}, i64 0, i64 3038287259199220266)
; CHECK:  ret void

define void @LargeAlloca() sanitize_memtag {
entry:
  %x = alloca i32, i32 256, align 16
  %0 = bitcast i32* %x to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %0, i8 42, i64 256, i1 false)
  call void @use(i8* nonnull %0)
  ret void
}

; CHECK-LABEL: define void @LargeAlloca(
; CHECK:  call void @llvm.aarch64.settag(i8* {{.*}}, i64 1024)
; CHECK:  call void @llvm.memset.p0i8.i64(i8* {{.*}}, i8 42, i64 256,
; CHECK:  ret void
