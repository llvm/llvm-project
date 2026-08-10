; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define i64 @test_vaddlv_s32(<2 x i32> %a1) nounwind readnone {
; CHECK: test_vaddlv_s32
; CHECK: saddlp.1d v[[REGNUM:[0-9]+]], v[[INREG:[0-9]+]]
; CHECK-NEXT: fmov x[[OUTREG:[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddlv.i = tail call i64 @llvm.aarch64.neon.saddlv.i64.v2i32(<2 x i32> %a1) nounwind
  ret i64 %vaddlv.i
}

define i64 @test_vaddlv_u32(<2 x i32> %a1) nounwind readnone {
; CHECK: test_vaddlv_u32
; CHECK: uaddlp.1d v[[REGNUM:[0-9]+]], v[[INREG:[0-9]+]]
; CHECK-NEXT: fmov x[[OUTREG:[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddlv.i = tail call i64 @llvm.aarch64.neon.uaddlv.i64.v2i32(<2 x i32> %a1) nounwind
  ret i64 %vaddlv.i
}

declare i64 @llvm.aarch64.neon.uaddlv.i64.v2i32(<2 x i32>) nounwind readnone

declare i64 @llvm.aarch64.neon.saddlv.i64.v2i32(<2 x i32>) nounwind readnone

