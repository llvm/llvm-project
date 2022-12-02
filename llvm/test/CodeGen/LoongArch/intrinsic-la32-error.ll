; RUN: not llc --mtriple=loongarch32 --disable-verify < %s 2>&1 | FileCheck %s

declare i32 @llvm.loongarch.crc.w.b.w(i32, i32)
declare i32 @llvm.loongarch.crc.w.h.w(i32, i32)
declare i32 @llvm.loongarch.crc.w.w.w(i32, i32)
declare i32 @llvm.loongarch.crc.w.d.w(i64, i32)
declare i32 @llvm.loongarch.crcc.w.b.w(i32, i32)
declare i32 @llvm.loongarch.crcc.w.h.w(i32, i32)
declare i32 @llvm.loongarch.crcc.w.w.w(i32, i32)
declare i32 @llvm.loongarch.crcc.w.d.w(i64, i32)

define i32 @crc_w_b_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.b.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.b.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crc_w_h_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.h.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.h.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crc_w_w_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.w.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.w.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crc_w_d_w(i64 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.d.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.d.w(i64 %a, i32 %b)
  ret i32 %res
}

define i32 @crcc_w_b_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crcc.w.b.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crcc.w.b.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crcc_w_h_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crcc.w.h.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crcc.w.h.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crcc_w_w_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crcc.w.w.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crcc.w.w.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crcc_w_d_w(i64 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crcc.w.d.w requires target: loongarch64
entry:
  %res = call i32 @llvm.loongarch.crcc.w.d.w(i64 %a, i32 %b)
  ret i32 %res
}
