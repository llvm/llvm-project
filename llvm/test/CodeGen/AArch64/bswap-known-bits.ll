; RUN: llc < %s -mtriple=aarch64-apple-darwin  | FileCheck %s

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)

; CHECK-LABEL: @test1
; CHECK: orr w0, wzr, #0x1
define i1 @test1(i16 %arg) {
  %a = or i16 %arg, 511
  %b = call i16 @llvm.bswap.i16(i16 %a)
  %and = and i16 %b, 256
  %res = icmp eq i16 %and, 256
  ret i1 %res
}

; CHECK-LABEL: @test2
; CHECK: orr w0, wzr, #0x1
define i1 @test2(i16 %arg) {
  %a = or i16 %arg, 1
  %b = call i16 @llvm.bswap.i16(i16 %a)
  %and = and i16 %b, 256
  %res = icmp eq i16 %and, 256
  ret i1 %res
}

; CHECK-LABEL: @test3
; CHECK: orr w0, wzr, #0x1
define i1 @test3(i16 %arg) {
  %a = or i16 %arg, 256
  %b = call i16 @llvm.bswap.i16(i16 %a)
  %and = and i16 %b, 1
  %res = icmp eq i16 %and, 1
  ret i1 %res
}

; CHECK-LABEL: @test4
; CHECK: orr w0, wzr, #0x1
define i1 @test4(i32 %arg) {
  %a = or i32 %arg, 2147483647  ; i32_MAX
  %b = call i32 @llvm.bswap.i32(i32 %a)
  %and = and i32 %b, 127
  %res = icmp eq i32 %and, 127
  ret i1 %res
}
