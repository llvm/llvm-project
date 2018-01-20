; RUN: opt < %s -basicaa -dse -S | FileCheck %s

define void @write4to7(i32* nocapture %p) {
; CHECK-LABEL: @write4to7(
entry:
  %arrayidx0 = getelementptr inbounds i32, i32* %p, i64 1
  %p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds i8, i8* %p3, i64 4
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 [[GEP]], i8 0, i64 24, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 28, i1 false)
  %arrayidx1 = getelementptr inbounds i32, i32* %p, i64 1
  store i32 1, i32* %arrayidx1, align 4
  ret void
}

define void @write0to3(i32* nocapture %p) {
; CHECK-LABEL: @write0to3(
entry:
  %p3 = bitcast i32* %p to i8*
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds i8, i8* %p3, i64 4
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 [[GEP]], i8 0, i64 24, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 28, i1 false)
  store i32 1, i32* %p, align 4
  ret void
}

define void @write0to7(i32* nocapture %p) {
; CHECK-LABEL: @write0to7(
entry:
  %p3 = bitcast i32* %p to i8*
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds i8, i8* %p3, i64 8
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 [[GEP]], i8 0, i64 24, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 32, i1 false)
  %p4 = bitcast i32* %p to i64*
  store i64 1, i64* %p4, align 8
  ret void
}

define void @write0to7_2(i32* nocapture %p) {
; CHECK-LABEL: @write0to7_2(
entry:
  %arrayidx0 = getelementptr inbounds i32, i32* %p, i64 1
  %p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds i8, i8* %p3, i64 4
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 [[GEP]], i8 0, i64 24, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 28, i1 false)
  %p4 = bitcast i32* %p to i64*
  store i64 1, i64* %p4, align 8
  ret void
}

; We do not trim the beginning of the eariler write if the alignment of the
; start pointer is changed.
define void @dontwrite0to3_align8(i32* nocapture %p) {
; CHECK-LABEL: @dontwrite0to3_align8(
entry:
  %p3 = bitcast i32* %p to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 8 %p3, i8 0, i64 32, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 8 %p3, i8 0, i64 32, i1 false)
  store i32 1, i32* %p, align 4
  ret void
}

define void @dontwrite0to1(i32* nocapture %p) {
; CHECK-LABEL: @dontwrite0to1(
entry:
  %p3 = bitcast i32* %p to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 32, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 32, i1 false)
  %p4 = bitcast i32* %p to i16*
  store i16 1, i16* %p4, align 4
  ret void
}

define void @dontwrite2to9(i32* nocapture %p) {
; CHECK-LABEL: @dontwrite2to9(
entry:
  %arrayidx0 = getelementptr inbounds i32, i32* %p, i64 1
  %p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 32, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 32, i1 false)
  %p4 = bitcast i32* %p to i16*
  %arrayidx2 = getelementptr inbounds i16, i16* %p4, i64 1
  %p5 = bitcast i16* %arrayidx2 to i64*
  store i64 1, i64* %p5, align 8
  ret void
}

define void @write8To15AndThen0To7(i64* nocapture %P) {
entry:
; CHECK-LABEL: @write8To15AndThen0To7(
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds i8, i8* %mybase0, i64 16
; CHECK: tail call void @llvm.memset.p0i8.i64(i8* align 8 [[GEP]], i8 0, i64 16, i1 false)

  %base0 = bitcast i64* %P to i8*
  %mybase0 = getelementptr inbounds i8, i8* %base0, i64 0
  tail call void @llvm.memset.p0i8.i64(i8* align 8 %mybase0, i8 0, i64 32, i1 false)

  %base64_0 = getelementptr inbounds i64, i64* %P, i64 0
  %base64_1 = getelementptr inbounds i64, i64* %P, i64 1

  store i64 1, i64* %base64_1
  store i64 2, i64* %base64_0
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind

