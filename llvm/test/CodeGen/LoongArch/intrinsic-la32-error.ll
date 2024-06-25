; RUN: not llc --mtriple=loongarch32 -mattr=+d < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.cacop.w(i32, i32, i32)
declare i32 @llvm.loongarch.crc.w.b.w(i32, i32)
declare i32 @llvm.loongarch.crc.w.h.w(i32, i32)
declare i32 @llvm.loongarch.crc.w.w.w(i32, i32)
declare i32 @llvm.loongarch.crc.w.d.w(i64, i32)
declare i32 @llvm.loongarch.crcc.w.b.w(i32, i32)
declare i32 @llvm.loongarch.crcc.w.h.w(i32, i32)
declare i32 @llvm.loongarch.crcc.w.w.w(i32, i32)
declare i32 @llvm.loongarch.crcc.w.d.w(i64, i32)
declare i64 @llvm.loongarch.csrrd.d(i32 immarg)
declare i64 @llvm.loongarch.csrwr.d(i64, i32 immarg)
declare i64 @llvm.loongarch.csrxchg.d(i64, i64, i32 immarg)
declare i64 @llvm.loongarch.iocsrrd.d(i32)
declare void @llvm.loongarch.iocsrwr.d(i64, i32)
declare void @llvm.loongarch.asrtle.d(i64, i64)
declare void @llvm.loongarch.asrtgt.d(i64, i64)
declare i64 @llvm.loongarch.lddir.d(i64, i64 immarg)
declare void @llvm.loongarch.ldpte.d(i64, i64 immarg)

define void @cacop_arg0_out_of_hi_range(i32 %a) nounwind {
; CHECK: llvm.loongarch.cacop.w: argument out of range
entry:
  call void @llvm.loongarch.cacop.w(i32 32, i32 %a, i32 1024)
  ret void
}

define void @cacop_arg0_out_of_lo_range(i32 %a) nounwind {
; CHECK: llvm.loongarch.cacop.w: argument out of range
entry:
  call void @llvm.loongarch.cacop.w(i32 -1, i32 %a, i32 1024)
  ret void
}

define void @cacop_arg2_out_of_hi_range(i32 %a) nounwind {
; CHECK: llvm.loongarch.cacop.w: argument out of range
entry:
  call void @llvm.loongarch.cacop.w(i32 1, i32 %a, i32 4096)
  ret void
}

define void @cacop_arg2_out_of_lo_range(i32 %a) nounwind {
; CHECK: llvm.loongarch.cacop.w: argument out of range
entry:
  call void @llvm.loongarch.cacop.w(i32 1, i32 %a, i32 -4096)
  ret void
}

define i32 @crc_w_b_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.b.w: requires loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.b.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crc_w_h_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.h.w: requires loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.h.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crc_w_w_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.w.w: requires loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.w.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crc_w_d_w(i64 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crc.w.d.w: requires loongarch64
entry:
  %res = call i32 @llvm.loongarch.crc.w.d.w(i64 %a, i32 %b)
  ret i32 %res
}

define i32 @crcc_w_b_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crcc.w.b.w: requires loongarch64
entry:
  %res = call i32 @llvm.loongarch.crcc.w.b.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crcc_w_h_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crcc.w.h.w: requires loongarch64
entry:
  %res = call i32 @llvm.loongarch.crcc.w.h.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crcc_w_w_w(i32 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crcc.w.w.w: requires loongarch64
entry:
  %res = call i32 @llvm.loongarch.crcc.w.w.w(i32 %a, i32 %b)
  ret i32 %res
}

define i32 @crcc_w_d_w(i64 %a, i32 %b) nounwind {
; CHECK: llvm.loongarch.crcc.w.d.w: requires loongarch64
entry:
  %res = call i32 @llvm.loongarch.crcc.w.d.w(i64 %a, i32 %b)
  ret i32 %res
}

define i64 @csrrd_d() {
; CHECK: llvm.loongarch.csrrd.d: requires loongarch64
entry:
  %0 = tail call i64 @llvm.loongarch.csrrd.d(i32 1)
  ret i64 %0
}

define i64 @csrwr_d(i64 %a) {
; CHECK: llvm.loongarch.csrwr.d: requires loongarch64
entry:
  %0 = tail call i64 @llvm.loongarch.csrwr.d(i64 %a, i32 1)
  ret i64 %0
}

define i64 @csrxchg_d(i64 %a, i64 %b) {
; CHECK: llvm.loongarch.csrxchg.d: requires loongarch64
entry:
  %0 = tail call i64 @llvm.loongarch.csrxchg.d(i64 %a, i64 %b, i32 1)
  ret i64 %0
}

define i64 @iocsrrd_d(i32 %a) {
; CHECK: llvm.loongarch.iocsrrd.d: requires loongarch64
entry:
  %0 = tail call i64 @llvm.loongarch.iocsrrd.d(i32 %a)
  ret i64 %0
}

define void @iocsrwr_d(i64 %a, i32 signext %b) {
; CHECK: llvm.loongarch.iocsrwr.d: requires loongarch64
entry:
  tail call void @llvm.loongarch.iocsrwr.d(i64 %a, i32 %b)
  ret void
}

define void @asrtle_d(i64 %a, i64 %b) {
; CHECK: llvm.loongarch.asrtle.d: requires loongarch64
entry:
  tail call void @llvm.loongarch.asrtle.d(i64 %a, i64 %b)
  ret void
}

define void @asrtgt_d(i64 %a, i64 %b) {
; CHECK: llvm.loongarch.asrtgt.d: requires loongarch64
entry:
  tail call void @llvm.loongarch.asrtgt.d(i64 %a, i64 %b)
  ret void
}

define i64 @lddir_d(i64 %a) {
; CHECK: llvm.loongarch.lddir.d: requires loongarch64
entry:
  %0 = tail call i64 @llvm.loongarch.lddir.d(i64 %a, i64 1)
  ret i64 %0
}

define void @ldpte_d(i64 %a) {
; CHECK: llvm.loongarch.ldpte.d: requires loongarch64
entry:
  tail call void @llvm.loongarch.ldpte.d(i64 %a, i64 1)
  ret void
}
