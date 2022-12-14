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

declare i32 @llvm.objectsize.i32(ptr, i1) nounwind readonly
define i32 @test.objectsize() {
; CHECK-LABEL: @test.objectsize(
; CHECK: @llvm.objectsize.i32.p0(ptr @a, i1 false, i1 false, i1 false)
  %s = call i32 @llvm.objectsize.i32(ptr @a, i1 false)
  ret i32 %s
}

declare i64 @llvm.objectsize.i64.p0(ptr, i1) nounwind readonly
define i64 @test.objectsize.2() {
; CHECK-LABEL: @test.objectsize.2(
; CHECK: @llvm.objectsize.i64.p0(ptr @a, i1 false, i1 false, i1 false)
  %s = call i64 @llvm.objectsize.i64.p0(ptr @a, i1 false)
  ret i64 %s
}

@u = private global [60 x ptr] zeroinitializer, align 1

declare i32 @llvm.objectsize.i32.unnamed(ptr, i1) nounwind readonly
define i32 @test.objectsize.unnamed() {
; CHECK-LABEL: @test.objectsize.unnamed(
; CHECK: @llvm.objectsize.i32.p0(ptr @u, i1 false, i1 false, i1 false)
  %s = call i32 @llvm.objectsize.i32.unnamed(ptr @u, i1 false)
  ret i32 %s
}

define i64 @test.objectsize.unnamed.2() {
; CHECK-LABEL: @test.objectsize.unnamed.2(
; CHECK: @llvm.objectsize.i64.p0(ptr @u, i1 false, i1 false, i1 false)
  %s = call i64 @llvm.objectsize.i64.p0(ptr @u, i1 false)
  ret i64 %s
}

declare <2 x double> @llvm.masked.load.v2f64(ptr %ptrs, i32, <2 x i1> %mask, <2 x double> %src0)

define <2 x double> @tests.masked.load(ptr %ptr, <2 x i1> %mask, <2 x double> %passthru)  {
; CHECK-LABEL: @tests.masked.load(
; CHECK: @llvm.masked.load.v2f64.p0
  %res = call <2 x double> @llvm.masked.load.v2f64(ptr %ptr, i32 1, <2 x i1> %mask, <2 x double> %passthru)
  ret <2 x double> %res
}

declare void @llvm.masked.store.v2f64(<2 x double> %val, ptr %ptrs, i32, <2 x i1> %mask)

define void @tests.masked.store(ptr %ptr, <2 x i1> %mask, <2 x double> %val)  {
; CHECK-LABEL: @tests.masked.store(
; CHECK: @llvm.masked.store.v2f64.p0
  call void @llvm.masked.store.v2f64(<2 x double> %val, ptr %ptr, i32 4, <2 x i1> %mask)
  ret void
}

declare <2 x double> @llvm.masked.gather.v2f64(<2 x ptr> %ptrs, i32, <2 x i1> %mask, <2 x double> %src0)

define <2 x double> @tests.masked.gather(<2 x ptr> %ptr, <2 x i1> %mask, <2 x double> %passthru)  {
; CHECK-LABEL: @tests.masked.gather(
; CHECK: @llvm.masked.gather.v2f64.v2p0
  %res = call <2 x double> @llvm.masked.gather.v2f64(<2 x ptr> %ptr, i32 1, <2 x i1> %mask, <2 x double> %passthru)
  ret <2 x double> %res
}

declare void @llvm.masked.scatter.v2f64(<2 x double> %val, <2 x ptr> %ptrs, i32, <2 x i1> %mask)

define void @tests.masked.scatter(<2 x ptr> %ptr, <2 x i1> %mask, <2 x double> %val)  {
; CHECK-LABEL: @tests.masked.scatter(
; CHECK: @llvm.masked.scatter.v2f64.v2p0
  call void @llvm.masked.scatter.v2f64(<2 x double> %val, <2 x ptr> %ptr, i32 1, <2 x i1> %mask)
  ret void
}

declare ptr @llvm.invariant.start(i64, ptr nocapture) nounwind readonly
declare void @llvm.invariant.end(ptr, i64, ptr nocapture) nounwind

define void @tests.invariant.start.end() {
  ; CHECK-LABEL: @tests.invariant.start.end(
  %a = alloca i8
  %i = call ptr @llvm.invariant.start(i64 1, ptr %a)
  ; CHECK: call ptr @llvm.invariant.start.p0
  store i8 0, ptr %a
  call void @llvm.invariant.end(ptr %i, i64 1, ptr %a)
  ; CHECK: call void @llvm.invariant.end.p0
  ret void
}

declare ptr @llvm.invariant.start.unnamed(i64, ptr nocapture) nounwind readonly
declare void @llvm.invariant.end.unnamed(ptr, i64, ptr nocapture) nounwind

define void @tests.invariant.start.end.unnamed() {
  ; CHECK-LABEL: @tests.invariant.start.end.unnamed(
  %a = alloca ptr
  %i = call ptr @llvm.invariant.start.unnamed(i64 1, ptr %a)
  ; CHECK: call ptr @llvm.invariant.start.p0
  store ptr null, ptr %a
  call void @llvm.invariant.end.unnamed(ptr %i, i64 1, ptr %a)
  ; CHECK: call void @llvm.invariant.end.p0
  ret void
}

@__stack_chk_guard = external global ptr
declare void @llvm.stackprotectorcheck(ptr)

define void @test.stackprotectorcheck() {
; CHECK-LABEL: @test.stackprotectorcheck(
; CHECK-NEXT: ret void
  call void @llvm.stackprotectorcheck(ptr @__stack_chk_guard)
  ret void
}

declare void  @llvm.lifetime.start(i64, ptr nocapture) nounwind readonly
declare void @llvm.lifetime.end(i64, ptr nocapture) nounwind

define void @tests.lifetime.start.end() {
  ; CHECK-LABEL: @tests.lifetime.start.end(
  %a = alloca i8
  call void @llvm.lifetime.start(i64 1, ptr %a)
  ; CHECK: call void @llvm.lifetime.start.p0(i64 1, ptr %a)
  store i8 0, ptr %a
  call void @llvm.lifetime.end(i64 1, ptr %a)
  ; CHECK: call void @llvm.lifetime.end.p0(i64 1, ptr %a)
  ret void
}

declare void  @llvm.lifetime.start.unnamed(i64, ptr nocapture) nounwind readonly
declare void @llvm.lifetime.end.unnamed(i64, ptr nocapture) nounwind

define void @tests.lifetime.start.end.unnamed() {
  ; CHECK-LABEL: @tests.lifetime.start.end.unnamed(
  %a = alloca ptr
  call void @llvm.lifetime.start.unnamed(i64 1, ptr %a)
  ; CHECK: call void @llvm.lifetime.start.p0(i64 1, ptr %a)
  store ptr null, ptr %a
  call void @llvm.lifetime.end.unnamed(i64 1, ptr %a)
  ; CHECK: call void @llvm.lifetime.end.p0(i64 1, ptr %a)
  ret void
}

declare void @llvm.prefetch(ptr, i32, i32, i32)
define void @test.prefetch(ptr %ptr) {
; CHECK-LABEL: @test.prefetch(
; CHECK: @llvm.prefetch.p0(ptr %ptr, i32 0, i32 3, i32 1)
  call void @llvm.prefetch(ptr %ptr, i32 0, i32 3, i32 1)
  ret void
}

declare void @llvm.prefetch.p0(ptr, i32, i32, i32)
define void @test.prefetch.2(ptr %ptr) {
; CHECK-LABEL: @test.prefetch.2(
; CHECK: @llvm.prefetch.p0(ptr %ptr, i32 0, i32 3, i32 1)
  call void @llvm.prefetch(ptr %ptr, i32 0, i32 3, i32 1)
  ret void
}

declare void @llvm.prefetch.unnamed(ptr, i32, i32, i32)
define void @test.prefetch.unnamed(ptr %ptr) {
; CHECK-LABEL: @test.prefetch.unnamed(
; CHECK: @llvm.prefetch.p0(ptr %ptr, i32 0, i32 3, i32 1)
  call void @llvm.prefetch.unnamed(ptr %ptr, i32 0, i32 3, i32 1)
  ret void
}

; This is part of @test.objectsize(), since llvm.objectsize declaration gets
; emitted at the end.
; CHECK: declare i32 @llvm.objectsize.i32.p0

; CHECK: declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
; CHECK: declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
