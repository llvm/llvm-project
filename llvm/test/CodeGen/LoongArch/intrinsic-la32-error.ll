; RUN: not llc --mtriple=loongarch32 --disable-verify < %s 2>&1 | FileCheck %s

declare i32 @llvm.loongarch.crc.w.d.w(i64, i32)

define i32 @crc_w_d_w(i64 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.d.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.d.w(i64 %a, i32 %b)
  ret i32 %res
}
