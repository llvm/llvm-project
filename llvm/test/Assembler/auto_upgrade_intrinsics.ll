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
; CHECK-DAG: declare i32 @llvm.objectsize.i32.p0i8
  %s = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i1 false)
  ret i32 %s
}
