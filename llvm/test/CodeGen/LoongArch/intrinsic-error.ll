; RUN: not llc --mtriple=loongarch32 < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.dbar(i32)
declare void @llvm.loongarch.ibar(i32)
declare void @llvm.loongarch.break(i32)
declare void @llvm.loongarch.movgr2fcsr(i32, i32)
declare i32 @llvm.loongarch.movfcsr2gr(i32)
declare void @llvm.loongarch.syscall(i32)
declare i32 @llvm.loongarch.csrrd.w(i32 immarg)
declare i32 @llvm.loongarch.csrwr.w(i32, i32 immarg)
declare i32 @llvm.loongarch.csrxchg.w(i32, i32, i32 immarg)

define void @dbar_imm_out_of_hi_range() #0 {
; CHECK: llvm.loongarch.dbar: argument out of range.
entry:
  call void @llvm.loongarch.dbar(i32 32769)
  ret void
}

define void @dbar_imm_out_of_lo_range() #0 {
; CHECK: llvm.loongarch.dbar: argument out of range.
entry:
  call void @llvm.loongarch.dbar(i32 -1)
  ret void
}

define void @ibar_imm_out_of_hi_range() #0 {
; CHECK: llvm.loongarch.ibar: argument out of range.
entry:
  call void @llvm.loongarch.ibar(i32 32769)
  ret void
}

define void @ibar_imm_out_of_lo_range() #0 {
; CHECK: llvm.loongarch.ibar: argument out of range.
entry:
  call void @llvm.loongarch.ibar(i32 -1)
  ret void
}

define void @break_imm_out_of_hi_range() #0 {
; CHECK: llvm.loongarch.break: argument out of range.
entry:
  call void @llvm.loongarch.break(i32 32769)
  ret void
}

define void @break_imm_out_of_lo_range() #0 {
; CHECK: llvm.loongarch.break: argument out of range.
entry:
  call void @llvm.loongarch.break(i32 -1)
  ret void
}

define void @movgr2fcsr(i32 %a) nounwind {
; CHECK: llvm.loongarch.movgr2fcsr: requires basic 'f' target feature.
entry:
  call void @llvm.loongarch.movgr2fcsr(i32 1, i32 %a)
  ret void
}

define void @movgr2fcsr_imm_out_of_hi_range(i32 %a) #0 {
; CHECK: llvm.loongarch.movgr2fcsr: argument out of range.
entry:
  call void @llvm.loongarch.movgr2fcsr(i32 32, i32 %a)
  ret void
}

define void @movgr2fcsr_imm_out_of_lo_range(i32 %a) #0 {
; CHECK: llvm.loongarch.movgr2fcsr: argument out of range.
entry:
  call void @llvm.loongarch.movgr2fcsr(i32 -1, i32 %a)
  ret void
}

define i32 @movfcsr2gr() nounwind {
; CHECK: llvm.loongarch.movfcsr2gr: requires basic 'f' target feature.
entry:
  %res = call i32 @llvm.loongarch.movfcsr2gr(i32 1)
  ret i32 %res
}

define i32 @movfcsr2gr_imm_out_of_hi_range() #0 {
; CHECK: llvm.loongarch.movfcsr2gr: argument out of range.
entry:
  %res = call i32 @llvm.loongarch.movfcsr2gr(i32 32)
  ret i32 %res
}

define i32 @movfcsr2gr_imm_out_of_lo_range() #0 {
; CHECK: llvm.loongarch.movfcsr2gr: argument out of range.
entry:
  %res = call i32 @llvm.loongarch.movfcsr2gr(i32 -1)
  ret i32 %res
}

define void @syscall_imm_out_of_hi_range() #0 {
; CHECK: llvm.loongarch.syscall: argument out of range.
entry:
  call void @llvm.loongarch.syscall(i32 32769)
  ret void
}

define void @syscall_imm_out_of_lo_range() #0 {
; CHECK: llvm.loongarch.syscall: argument out of range.
entry:
  call void @llvm.loongarch.syscall(i32 -1)
  ret void
}

define i32 @csrrd_w_imm_out_of_hi_range() #0 {
; CHECK: llvm.loongarch.csrrd.w: argument out of range.
entry:
  %0 = call i32 @llvm.loongarch.csrrd.w(i32 16384)
  ret i32 %0
}

define i32 @csrrd_w_imm_out_of_lo_range() #0 {
; CHECK: llvm.loongarch.csrrd.w: argument out of range.
entry:
  %0 = call i32 @llvm.loongarch.csrrd.w(i32 -1)
  ret i32 %0
}

define i32 @csrwr_w_imm_out_of_hi_range(i32 %a) #0 {
; CHECK: llvm.loongarch.csrwr.w: argument out of range.
entry:
  %0 = call i32 @llvm.loongarch.csrwr.w(i32 %a, i32 16384)
  ret i32 %0
}

define i32 @csrwr_w_imm_out_of_lo_range(i32 %a) #0 {
; CHECK: llvm.loongarch.csrwr.w: argument out of range.
entry:
  %0 = call i32 @llvm.loongarch.csrwr.w(i32 %a, i32 -1)
  ret i32 %0
}

define i32 @csrxchg_w_imm_out_of_hi_range(i32 %a, i32 %b) #0 {
; CHECK: llvm.loongarch.csrxchg.w: argument out of range.
entry:
  %0 = call i32 @llvm.loongarch.csrxchg.w(i32 %a, i32 %b, i32 16384)
  ret i32 %0
}

define i32 @csrxchg_w_imm_out_of_lo_range(i32 %a, i32 %b) #0 {
; CHECK: llvm.loongarch.csrxchg.w: argument out of range.
entry:
  %0 = call i32 @llvm.loongarch.csrxchg.w(i32 %a, i32 %b, i32 -1)
  ret i32 %0
}

attributes #0 = { nounwind "target-features"="+f" }
