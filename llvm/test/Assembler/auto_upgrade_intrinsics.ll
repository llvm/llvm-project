; Test to make sure intrinsics are automatically upgraded.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

declare i8 @llvm.ctlz.i8(i8)
declare i16 @llvm.ctlz.i16(i16)
declare i32 @llvm.ctlz.i32(i32)
declare i42 @llvm.ctlz.i42(i42)  ; Not a power-of-2


declare i32 @llvm.objectsize.i32(i8*, i1) nounwind readonly


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

define i32 @test.objectsize() {
; CHECK-LABEL: @test.objectsize(
; CHECK: @llvm.objectsize.i32.p0i8
  %s = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i1 false)
  ret i32 %s
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
  call void @llvm.masked.store.v2f64(<2 x double> %val, <2 x double>* %ptr, i32 3, <2 x i1> %mask)
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

; This is part of @test.objectsize(), since llvm.objectsize declaration gets
; emitted at the end.
; CHECK: declare i32 @llvm.objectsize.i32.p0i8
