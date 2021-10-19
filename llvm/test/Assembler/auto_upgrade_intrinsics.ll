; Test to make sure intrinsics are automatically upgraded.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

%0 = type opaque;

declare i8 @llvm.ctlz.i8(i8)
declare i16 @llvm.ctlz.i16(i16)
declare i32 @llvm.ctlz.i32(i32)
declare i42 @llvm.ctlz.i42(i42)  ; Not a power-of-2


define void @test.ctlz(i8 %a, i16 %b, i32 %c, i42 %d) {
; CHECK: @test.ctlz

entry:
  ; CHECK: call i8 @llvm.ctlz.i8(i8 %a, i1 false)
  call i8 @llvm.ctlz.i8(i8 %a)
  ; CHECK: call i16 @llvm.ctlz.i16(i16 %b, i1 false)
  call i16 @llvm.ctlz.i16(i16 %b)
  ; CHECK: call i32 @llvm.ctlz.i32(i32 %c, i1 false)
  call i32 @llvm.ctlz.i32(i32 %c)
  ; CHECK: call i42 @llvm.ctlz.i42(i42 %d, i1 false)
  call i42 @llvm.ctlz.i42(i42 %d)

  ret void
}

declare i8 @llvm.cttz.i8(i8)
declare i16 @llvm.cttz.i16(i16)
declare i32 @llvm.cttz.i32(i32)
declare i42 @llvm.cttz.i42(i42)  ; Not a power-of-2

define void @test.cttz(i8 %a, i16 %b, i32 %c, i42 %d) {
; CHECK: @test.cttz

entry:
  ; CHECK: call i8 @llvm.cttz.i8(i8 %a, i1 false)
  call i8 @llvm.cttz.i8(i8 %a)
  ; CHECK: call i16 @llvm.cttz.i16(i16 %b, i1 false)
  call i16 @llvm.cttz.i16(i16 %b)
  ; CHECK: call i32 @llvm.cttz.i32(i32 %c, i1 false)
  call i32 @llvm.cttz.i32(i32 %c)
  ; CHECK: call i42 @llvm.cttz.i42(i42 %d, i1 false)
  call i42 @llvm.cttz.i42(i42 %d)

  ret void
}


@a = private global [60 x i8] zeroinitializer, align 1

declare i32 @llvm.objectsize.i32(i8*, i1) nounwind readonly
define i32 @test.objectsize() {
; CHECK-LABEL: @test.objectsize(
; CHECK: @llvm.objectsize.i32.p0i8(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i1 false, i1 false, i1 false)
  %s = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i1 false)
  ret i32 %s
}

declare i64 @llvm.objectsize.i64.p0i8(i8*, i1) nounwind readonly
define i64 @test.objectsize.2() {
; CHECK-LABEL: @test.objectsize.2(
; CHECK: @llvm.objectsize.i64.p0i8(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i1 false, i1 false, i1 false)
  %s = call i64 @llvm.objectsize.i64.p0i8(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i1 false)
  ret i64 %s
}

@u = private global [60 x %0*] zeroinitializer, align 1

declare i32 @llvm.objectsize.i32.unnamed(%0**, i1) nounwind readonly
define i32 @test.objectsize.unnamed() {
; CHECK-LABEL: @test.objectsize.unnamed(
; CHECK: @llvm.objectsize.i32.p0p0s_s.0(%0** getelementptr inbounds ([60 x %0*], [60 x %0*]* @u, i32 0, i32 0), i1 false, i1 false, i1 false)
  %s = call i32 @llvm.objectsize.i32.unnamed(%0** getelementptr inbounds ([60 x %0*], [60 x %0*]* @u, i32 0, i32 0), i1 false)
  ret i32 %s
}

declare i64 @llvm.objectsize.i64.p0p0s_s.0(%0**, i1) nounwind readonly
define i64 @test.objectsize.unnamed.2() {
; CHECK-LABEL: @test.objectsize.unnamed.2(
; CHECK: @llvm.objectsize.i64.p0p0s_s.0(%0** getelementptr inbounds ([60 x %0*], [60 x %0*]* @u, i32 0, i32 0), i1 false, i1 false, i1 false)
  %s = call i64 @llvm.objectsize.i64.p0p0s_s.0(%0** getelementptr inbounds ([60 x %0*], [60 x %0*]* @u, i32 0, i32 0), i1 false)
  ret i64 %s
}

declare <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptrs, i32, <2 x i1> %mask, <2 x double> %src0)

define <2 x double> @tests.masked.load(<2 x double>* %ptr, <2 x i1> %mask, <2 x double> %passthru)  {
; CHECK-LABEL: @tests.masked.load(
; CHECK: @llvm.masked.load.v2f64.p0v2f64
  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 1, <2 x i1> %mask, <2 x double> %passthru)
  ret <2 x double> %res
}

declare void @llvm.masked.store.v2f64(<2 x double> %val, <2 x double>* %ptrs, i32, <2 x i1> %mask)

define void @tests.masked.store(<2 x double>* %ptr, <2 x i1> %mask, <2 x double> %val)  {
; CHECK-LABEL: @tests.masked.store(
; CHECK: @llvm.masked.store.v2f64.p0v2f64
  call void @llvm.masked.store.v2f64(<2 x double> %val, <2 x double>* %ptr, i32 4, <2 x i1> %mask)
  ret void
}

declare <2 x double> @llvm.masked.gather.v2f64(<2 x double*> %ptrs, i32, <2 x i1> %mask, <2 x double> %src0)

define <2 x double> @tests.masked.gather(<2 x double*> %ptr, <2 x i1> %mask, <2 x double> %passthru)  {
; CHECK-LABEL: @tests.masked.gather(
; CHECK: @llvm.masked.gather.v2f64.v2p0f64
  %res = call <2 x double> @llvm.masked.gather.v2f64(<2 x double*> %ptr, i32 1, <2 x i1> %mask, <2 x double> %passthru)
  ret <2 x double> %res
}

declare void @llvm.masked.scatter.v2f64(<2 x double> %val, <2 x double*> %ptrs, i32, <2 x i1> %mask)

define void @tests.masked.scatter(<2 x double*> %ptr, <2 x i1> %mask, <2 x double> %val)  {
; CHECK-LABEL: @tests.masked.scatter(
; CHECK: @llvm.masked.scatter.v2f64.v2p0f64
  call void @llvm.masked.scatter.v2f64(<2 x double> %val, <2 x double*> %ptr, i32 1, <2 x i1> %mask)
  ret void
}

declare {}* @llvm.invariant.start(i64, i8* nocapture) nounwind readonly
declare void @llvm.invariant.end({}*, i64, i8* nocapture) nounwind

define void @tests.invariant.start.end() {
  ; CHECK-LABEL: @tests.invariant.start.end(
  %a = alloca i8
  %i = call {}* @llvm.invariant.start(i64 1, i8* %a)
  ; CHECK: call {}* @llvm.invariant.start.p0i8
  store i8 0, i8* %a
  call void @llvm.invariant.end({}* %i, i64 1, i8* %a)
  ; CHECK: call void @llvm.invariant.end.p0i8
  ret void
}

declare {}* @llvm.invariant.start.unnamed(i64, %0** nocapture) nounwind readonly
declare void @llvm.invariant.end.unnamed({}*, i64, %0** nocapture) nounwind

define void @tests.invariant.start.end.unnamed() {
  ; CHECK-LABEL: @tests.invariant.start.end.unnamed(
  %a = alloca %0*
  %i = call {}* @llvm.invariant.start.unnamed(i64 1, %0** %a)
  ; CHECK: call {}* @llvm.invariant.start.p0p0s_s.0
  store %0* null, %0** %a
  call void @llvm.invariant.end.unnamed({}* %i, i64 1, %0** %a)
  ; CHECK: call void @llvm.invariant.end.p0p0s_s.0
  ret void
}

@__stack_chk_guard = external global i8*
declare void @llvm.stackprotectorcheck(i8**)

define void @test.stackprotectorcheck() {
; CHECK-LABEL: @test.stackprotectorcheck(
; CHECK-NEXT: ret void
  call void @llvm.stackprotectorcheck(i8** @__stack_chk_guard)
  ret void
}

declare void  @llvm.lifetime.start(i64, i8* nocapture) nounwind readonly
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

define void @tests.lifetime.start.end() {
  ; CHECK-LABEL: @tests.lifetime.start.end(
  %a = alloca i8
  call void @llvm.lifetime.start(i64 1, i8* %a)
  ; CHECK: call void @llvm.lifetime.start.p0i8(i64 1, i8* %a)
  store i8 0, i8* %a
  call void @llvm.lifetime.end(i64 1, i8* %a)
  ; CHECK: call void @llvm.lifetime.end.p0i8(i64 1, i8* %a)
  ret void
}

declare void  @llvm.lifetime.start.unnamed(i64, %0** nocapture) nounwind readonly
declare void @llvm.lifetime.end.unnamed(i64, %0** nocapture) nounwind

define void @tests.lifetime.start.end.unnamed() {
  ; CHECK-LABEL: @tests.lifetime.start.end.unnamed(
  %a = alloca %0*
  call void @llvm.lifetime.start.unnamed(i64 1, %0** %a)
  ; CHECK: call void @llvm.lifetime.start.p0p0s_s.0(i64 1, %0** %a)
  store %0* null, %0** %a
  call void @llvm.lifetime.end.unnamed(i64 1, %0** %a)
  ; CHECK: call void @llvm.lifetime.end.p0p0s_s.0(i64 1, %0** %a)
  ret void
}

declare void @llvm.prefetch(i8*, i32, i32, i32)
define void @test.prefetch(i8* %ptr) {
; CHECK-LABEL: @test.prefetch(
; CHECK: @llvm.prefetch.p0i8(i8* %ptr, i32 0, i32 3, i32 2)
  call void @llvm.prefetch(i8* %ptr, i32 0, i32 3, i32 2)
  ret void
}

declare void @llvm.prefetch.p0i8(i8*, i32, i32, i32)
define void @test.prefetch.2(i8* %ptr) {
; CHECK-LABEL: @test.prefetch.2(
; CHECK: @llvm.prefetch.p0i8(i8* %ptr, i32 0, i32 3, i32 2)
  call void @llvm.prefetch(i8* %ptr, i32 0, i32 3, i32 2)
  ret void
}

declare void @llvm.prefetch.unnamed(%0**, i32, i32, i32)
define void @test.prefetch.unnamed(%0** %ptr) {
; CHECK-LABEL: @test.prefetch.unnamed(
; CHECK: @llvm.prefetch.p0p0s_s.0(%0** %ptr, i32 0, i32 3, i32 2)
  call void @llvm.prefetch.unnamed(%0** %ptr, i32 0, i32 3, i32 2)
  ret void
}

; This is part of @test.objectsize(), since llvm.objectsize declaration gets
; emitted at the end.
; CHECK: declare i32 @llvm.objectsize.i32.p0i8
; CHECK: declare i32 @llvm.objectsize.i32.p0p0s_s.0

; CHECK: declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
; CHECK: declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
; CHECK: declare void @llvm.lifetime.start.p0p0s_s.0(i64 immarg, %0** nocapture)
; CHECK: declare void @llvm.lifetime.end.p0p0s_s.0(i64 immarg, %0** nocapture)
