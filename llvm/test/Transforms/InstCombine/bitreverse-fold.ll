; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %p) {
; CHECK-LABEL: @test1
; CHECK-NEXT: ret i32 %p
  %a = call i32 @llvm.bitreverse.i32(i32 %p)
  %b = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %b
}

declare i32 @llvm.bitreverse.i32(i32) readnone
