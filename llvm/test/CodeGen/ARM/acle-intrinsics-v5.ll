; RUN: llc -O1 -mtriple=armv5te-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=armv6-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=armv7-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=thumbv7-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=thumbv6t2-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=thumbv7em-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=thumbv8m.main-none-none-eabi -mattr=+dsp %s -o - | FileCheck %s
define i32 @smulbb(i32 %a, i32 %b) {
; CHECK-LABEL: smulbb
; CHECK: smulbb r0, r0, r1
  %tmp = call i32 @llvm.arm.smulbb(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smulbt(i32 %a, i32 %b) {
; CHECK-LABEL: smulbt
; CHECK: smulbt r0, r0, r1
  %tmp = call i32 @llvm.arm.smulbt(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smultb(i32 %a, i32 %b) {
; CHECK-LABEL: smultb
; CHECK: smultb r0, r0, r1
  %tmp = call i32 @llvm.arm.smultb(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smultt(i32 %a, i32 %b) {
; CHECK-LABEL: smultt
; CHECK: smultt r0, r0, r1
  %tmp = call i32 @llvm.arm.smultt(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smulwb(i32 %a, i32 %b) {
; CHECK-LABEL: smulwb
; CHECK: smulwb r0, r0, r1
  %tmp = call i32 @llvm.arm.smulwb(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smulwt(i32 %a, i32 %b) {
; CHECK-LABEL: smulwt
; CHECK: smulwt r0, r0, r1
  %tmp = call i32 @llvm.arm.smulwt(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @acc_mults(i32 %a, i32 %b, i32 %acc) {
; CHECK-LABEL: acc_mults
; CHECK: smlabb r2, r0, r1, r2
; CHECK: smlabt r2, r0, r1, r2
; CHECK: smlatb r2, r0, r1, r2
; CHECK: smlatt r2, r0, r1, r2
; CHECK: smlawb r2, r0, r1, r2
; CHECK: smlawt r0, r0, r1, r2
  %acc1 = call i32 @llvm.arm.smlabb(i32 %a, i32 %b, i32 %acc)
  %acc2 = call i32 @llvm.arm.smlabt(i32 %a, i32 %b, i32 %acc1)
  %acc3 = call i32 @llvm.arm.smlatb(i32 %a, i32 %b, i32 %acc2)
  %acc4 = call i32 @llvm.arm.smlatt(i32 %a, i32 %b, i32 %acc3)
  %acc5 = call i32 @llvm.arm.smlawb(i32 %a, i32 %b, i32 %acc4)
  %acc6 = call i32 @llvm.arm.smlawt(i32 %a, i32 %b, i32 %acc5)
  ret i32 %acc6
}

define i32 @qadd(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qadd
; CHECK: qadd r0, r0, r1
  %tmp = call i32 @llvm.arm.qadd(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @qsub(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qsub
; CHECK: qsub r0, r0, r1
  %tmp = call i32 @llvm.arm.qsub(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @qdadd(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qdadd
; CHECK: qdadd r0, r1, r0
  %dbl = call i32 @llvm.arm.qadd(i32 %a, i32 %a)
  %add = call i32 @llvm.arm.qadd(i32 %dbl, i32 %b)
  ret i32 %add
}

define i32 @qdsub(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qdsub
; CHECK: qdsub r0, r0, r1
  %dbl = call i32 @llvm.arm.qadd(i32 %b, i32 %b)
  %add = call i32 @llvm.arm.qsub(i32 %a, i32 %dbl)
  ret i32 %add
}

declare i32 @llvm.arm.smulbb(i32 %a, i32 %b) nounwind readnone
declare i32 @llvm.arm.smulbt(i32 %a, i32 %b) nounwind readnone
declare i32 @llvm.arm.smultb(i32 %a, i32 %b) nounwind readnone
declare i32 @llvm.arm.smultt(i32 %a, i32 %b) nounwind readnone
declare i32 @llvm.arm.smulwb(i32 %a, i32 %b) nounwind readnone
declare i32 @llvm.arm.smulwt(i32 %a, i32 %b) nounwind readnone
declare i32 @llvm.arm.smlabb(i32, i32, i32) nounwind
declare i32 @llvm.arm.smlabt(i32, i32, i32) nounwind
declare i32 @llvm.arm.smlatb(i32, i32, i32) nounwind
declare i32 @llvm.arm.smlatt(i32, i32, i32) nounwind
declare i32 @llvm.arm.smlawb(i32, i32, i32) nounwind
declare i32 @llvm.arm.smlawt(i32, i32, i32) nounwind
declare i32 @llvm.arm.qadd(i32, i32) nounwind
declare i32 @llvm.arm.qsub(i32, i32) nounwind
