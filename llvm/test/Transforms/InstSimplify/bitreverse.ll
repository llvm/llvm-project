; RUN: opt < %s -S -instsimplify | FileCheck %s

declare i32 @llvm.bitreverse.i32(i32)

; CHECK-LABEL: @test1(
; CHECK: ret i1 false
define i1 @test1(i32 %arg) {
  %a = or i32 %arg, 1
  %b = call i32 @llvm.bitreverse.i32(i32 %a)
  %res = icmp eq i32 %b, 0
  ret i1 %res
}

; CHECK-LABEL: @test2(
; CHECK: ret i1 false
define i1 @test2(i32 %arg) {
  %a = or i32 %arg, 1024
  %b = call i32 @llvm.bitreverse.i32(i32 %a)
  %res = icmp eq i32 %b, 0
  ret i1 %res
}

; CHECK-LABEL: @test3(
; CHECK: ret i1 false
define i1 @test3(i32 %arg) {
  %a = and i32 %arg, 1
  %b = call i32 @llvm.bitreverse.i32(i32 %a)
  %and = and i32 %b, 1
  %res = icmp eq i32 %and, 1
  ret i1 %res
}
