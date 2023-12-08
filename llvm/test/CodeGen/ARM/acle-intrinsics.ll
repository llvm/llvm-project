; RUN: llc -O1 -mtriple=armv6-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=armv7-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=thumbv7-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=thumbv6t2-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=thumbv7em-none-none-eabi %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=thumbv8m.main-none-none-eabi -mattr=+dsp %s -o - | FileCheck %s


; upper-bound of the immediate argument
define i32 @ssat1(i32 %a) nounwind {
; CHECK-LABEL: ssat1
; CHECK: ssat r0, #32, r0
  %tmp = call i32 @llvm.arm.ssat(i32 %a, i32 32)
  ret i32 %tmp
}

; lower-bound of the immediate argument
define i32 @ssat2(i32 %a) nounwind {
; CHECK-LABEL: ssat2
; CHECK: ssat r0, #1, r0
  %tmp = call i32 @llvm.arm.ssat(i32 %a, i32 1)
  ret i32 %tmp
}

; upper-bound of the immediate argument
define i32 @usat1(i32 %a) nounwind {
; CHECK-LABEL: usat1
; CHECK: usat r0, #31, r0
  %tmp = call i32 @llvm.arm.usat(i32 %a, i32 31)
  ret i32 %tmp
}

; lower-bound of the immediate argument
define i32 @usat2(i32 %a) nounwind {
; CHECK-LABEL: usat2
; CHECK: usat r0, #0, r0
  %tmp = call i32 @llvm.arm.usat(i32 %a, i32 0)
  ret i32 %tmp
}

define i32 @ssat16 (i32 %a) nounwind {
; CHECK-LABEL: ssat16
; CHECK: ssat16 r0, #1, r0
; CHECK: ssat16 r0, #16, r0
  %tmp = call i32 @llvm.arm.ssat16(i32 %a, i32 1)
  %tmp2 = call i32 @llvm.arm.ssat16(i32 %tmp, i32 16)
  ret i32 %tmp2
}

define i32 @usat16(i32 %a) nounwind {
; CHECK-LABEL: usat16
; CHECK: usat16 r0, #0, r0
; CHECK: usat16 r0, #15, r0
  %tmp = call i32 @llvm.arm.usat16(i32 %a, i32 0)
  %tmp2 = call i32 @llvm.arm.usat16(i32 %tmp, i32 15)
  ret i32 %tmp2
}

define i32 @pack_unpack(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: pack_unpack
; CHECK: sxtab16 r0, r0, r1
; CHECK: sxtb16 r0, r0
; CHECK: uxtab16 r0, r1, r0
; CHECK: uxtb16 r0, r0
  %tmp = call i32 @llvm.arm.sxtab16(i32 %a, i32 %b)
  %tmp1 = call i32 @llvm.arm.sxtb16(i32 %tmp)
  %tmp2 = call i32 @llvm.arm.uxtab16(i32 %b, i32 %tmp1)
  %tmp3 = call i32 @llvm.arm.uxtb16(i32 %tmp2)
  ret i32 %tmp3
}

define i32 @sel(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: sel
; CHECK: sel r0, r0, r1
  %tmp = call i32 @llvm.arm.sel(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @qadd8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qadd8
; CHECK: qadd8 r0, r0, r1
  %tmp = call i32 @llvm.arm.qadd8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @qsub8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qsub8
; CHECK: qsub8 r0, r0, r1
  %tmp = call i32 @llvm.arm.qsub8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @sadd8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: sadd8
; CHECK: sadd8 r0, r0, r1
  %tmp = call i32 @llvm.arm.sadd8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @shadd8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: shadd8
; CHECK: shadd8 r0, r0, r1
  %tmp = call i32 @llvm.arm.shadd8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @shsub8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: shsub8
; CHECK: shsub8 r0, r0, r1
  %tmp = call i32 @llvm.arm.shsub8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @ssub8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: ssub8
; CHECK: ssub8 r0, r0, r1
  %tmp = call i32 @llvm.arm.ssub8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uadd8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uadd8
; CHECK: uadd8 r0, r0, r1
  %tmp = call i32 @llvm.arm.uadd8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uhadd8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uhadd8
; CHECK: uhadd8 r0, r0, r1
  %tmp = call i32 @llvm.arm.uhadd8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uhsub8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uhsub8
; CHECK: uhsub8 r0, r0, r1
  %tmp = call i32 @llvm.arm.uhsub8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uqadd8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uqadd8
; CHECK: uqadd8 r0, r0, r1
  %tmp = call i32 @llvm.arm.uqadd8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uqsub8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uqsub8
; CHECK: uqsub8 r0, r0, r1
  %tmp = call i32 @llvm.arm.uqsub8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @usub8(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: usub8
; CHECK: usub8 r0, r0, r1
  %tmp = call i32 @llvm.arm.usub8(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @usad(i32 %a, i32 %b, i32 %c) nounwind {
; CHECK-LABEL: usad
; CHECK: usad8 r0, r0, r1
; CHECK: usada8 r0, r0, r1, r2
  %tmp = call i32 @llvm.arm.usad8(i32 %a, i32 %b)
  %tmp1 = call i32 @llvm.arm.usada8(i32 %tmp, i32 %b, i32 %c)
  ret i32 %tmp1
}

define i32 @qadd16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qadd16
; CHECK: qadd16 r0, r0, r1
  %tmp = call i32 @llvm.arm.qadd16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @qasx(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qasx
; CHECK: qasx r0, r0, r1
  %tmp = call i32 @llvm.arm.qasx(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @qsax(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qsax
; CHECK: qsax r0, r0, r1
  %tmp = call i32 @llvm.arm.qsax(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @qsub16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: qsub16
; CHECK: qsub16 r0, r0, r1
  %tmp = call i32 @llvm.arm.qsub16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @sadd16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: sadd16
; CHECK: sadd16 r0, r0, r1
  %tmp = call i32 @llvm.arm.sadd16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @sasx(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: sasx
; CHECK: sasx r0, r0, r1
  %tmp = call i32 @llvm.arm.sasx(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @shadd16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: shadd16
; CHECK: shadd16 r0, r0, r1
  %tmp = call i32 @llvm.arm.shadd16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @shasx(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: shasx
; CHECK: shasx r0, r0, r1
  %tmp = call i32 @llvm.arm.shasx(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @shsax(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: shsax
; CHECK: shsax r0, r0, r1
  %tmp = call i32 @llvm.arm.shsax(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @shsub16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: shsub16
; CHECK: shsub16 r0, r0, r1
  %tmp = call i32 @llvm.arm.shsub16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @ssax(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: ssax
; CHECK: ssax r0, r0, r1
  %tmp = call i32 @llvm.arm.ssax(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @ssub16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: ssub16
; CHECK: ssub16 r0, r0, r1
  %tmp = call i32 @llvm.arm.ssub16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uadd16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uadd16
; CHECK: uadd16 r0, r0, r1
  %tmp = call i32 @llvm.arm.uadd16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uasx(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uasx
; CHECK: uasx r0, r0, r1
  %tmp = call i32 @llvm.arm.uasx(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uhadd16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uhadd16
; CHECK: uhadd16 r0, r0, r1
  %tmp = call i32 @llvm.arm.uhadd16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uhasx(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uhasx
; CHECK: uhasx r0, r0, r1
  %tmp = call i32 @llvm.arm.uhasx(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uhsax(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uhsax
; CHECK: uhsax r0, r0, r1
  %tmp = call i32 @llvm.arm.uhsax(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uhsub16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uhsub16
; CHECK: uhsub16 r0, r0, r1
  %tmp = call i32 @llvm.arm.uhsub16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uqadd16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uqadd16
; CHECK: uqadd16 r0, r0, r1
  %tmp = call i32 @llvm.arm.uqadd16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uqasx(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uqasx
; CHECK: uqasx r0, r0, r1
  %tmp = call i32 @llvm.arm.uqasx(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uqsax(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uqsax
; CHECK: uqsax r0, r0, r1
  %tmp = call i32 @llvm.arm.uqsax(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @uqsub16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: uqsub16
; CHECK: uqsub16 r0, r0, r1
  %tmp = call i32 @llvm.arm.uqsub16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @usax(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: usax
; CHECK: usax r0, r0, r1
  %tmp = call i32 @llvm.arm.usax(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @usub16(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: usub16
; CHECK: usub16 r0, r0, r1
  %tmp = call i32 @llvm.arm.usub16(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smlad(i32 %a, i32 %b, i32 %c) nounwind {
; CHECK-LABEL: smlad
; CHECK: smlad r0, r0, r1, r2
  %tmp = call i32 @llvm.arm.smlad(i32 %a, i32 %b, i32 %c)
  ret i32 %tmp
}

define i32 @smladx(i32 %a, i32 %b, i32 %c) nounwind {
; CHECK-LABEL: smladx
; CHECK: smladx r0, r0, r1, r2
  %tmp = call i32 @llvm.arm.smladx(i32 %a, i32 %b, i32 %c)
  ret i32 %tmp
}

define i64 @smlald(i32 %a, i32 %b, i64 %c) nounwind {
; CHECK-LABEL: smlald
; CHECK: smlald r2, r3, r0, r1
  %tmp = call i64 @llvm.arm.smlald(i32 %a, i32 %b, i64 %c)
  ret i64 %tmp
}

define i64 @smlaldx(i32 %a, i32 %b, i64 %c) nounwind {
; CHECK-LABEL: smlaldx
; CHECK: smlaldx r2, r3, r0, r1
  %tmp = call i64 @llvm.arm.smlaldx(i32 %a, i32 %b, i64 %c)
  ret i64 %tmp
}

define i32 @smlsd(i32 %a, i32 %b, i32 %c) nounwind {
; CHECK-LABEL: smlsd
; CHECK: smlsd r0, r0, r1, r2
  %tmp = call i32 @llvm.arm.smlsd(i32 %a, i32 %b, i32 %c)
  ret i32 %tmp
}

define i32 @smlsdx(i32 %a, i32 %b, i32 %c) nounwind {
; CHECK-LABEL: smlsdx
; CHECK: smlsdx r0, r0, r1, r2
  %tmp = call i32 @llvm.arm.smlsdx(i32 %a, i32 %b, i32 %c)
  ret i32 %tmp
}

define i64 @smlsld(i32 %a, i32 %b, i64 %c) nounwind {
; CHECK-LABEL: smlsld
; CHECK: smlsld r2, r3, r0, r1
  %tmp = call i64 @llvm.arm.smlsld(i32 %a, i32 %b, i64 %c)
  ret i64 %tmp
}

define i64 @smlsldx(i32 %a, i32 %b, i64 %c) nounwind {
; CHECK-LABEL: smlsldx
; CHECK: smlsldx r2, r3, r0, r1
  %tmp = call i64 @llvm.arm.smlsldx(i32 %a, i32 %b, i64 %c)
  ret i64 %tmp
}

define i32 @smuad(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: smuad
; CHECK: smuad r0, r0, r1
  %tmp = call i32 @llvm.arm.smuad(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smuadx(i32 %a, i32 %b) nounwind {
;CHECK-LABEL: smuadx
; CHECK: smuadx r0, r0, r1
  %tmp = call i32 @llvm.arm.smuadx(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smusd(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: smusd
; CHECK: smusd r0, r0, r1
  %tmp = call i32 @llvm.arm.smusd(i32 %a, i32 %b)
  ret i32 %tmp
}

define i32 @smusdx(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: smusdx
; CHECK: smusdx r0, r0, r1
  %tmp = call i32 @llvm.arm.smusdx(i32 %a, i32 %b)
  ret i32 %tmp
}
declare i32 @llvm.arm.ssat(i32, i32) nounwind readnone
declare i32 @llvm.arm.usat(i32, i32) nounwind readnone
declare i32 @llvm.arm.ssat16(i32, i32) nounwind
declare i32 @llvm.arm.usat16(i32, i32) nounwind
declare i32 @llvm.arm.sxtab16(i32, i32)
declare i32 @llvm.arm.sxtb16(i32)
declare i32 @llvm.arm.uxtab16(i32, i32)
declare i32 @llvm.arm.uxtb16(i32)
declare i32 @llvm.arm.sel(i32, i32) nounwind
declare i32 @llvm.arm.qadd8(i32, i32) nounwind
declare i32 @llvm.arm.qsub8(i32, i32) nounwind
declare i32 @llvm.arm.sadd8(i32, i32) nounwind
declare i32 @llvm.arm.shadd8(i32, i32) nounwind
declare i32 @llvm.arm.shsub8(i32, i32) nounwind
declare i32 @llvm.arm.ssub8(i32, i32) nounwind
declare i32 @llvm.arm.uadd8(i32, i32) nounwind
declare i32 @llvm.arm.uhadd8(i32, i32) nounwind
declare i32 @llvm.arm.uhsub8(i32, i32) nounwind
declare i32 @llvm.arm.uqadd8(i32, i32) nounwind
declare i32 @llvm.arm.uqsub8(i32, i32) nounwind
declare i32 @llvm.arm.usub8(i32, i32) nounwind
declare i32 @llvm.arm.usad8(i32, i32) nounwind readnone
declare i32 @llvm.arm.usada8(i32, i32, i32) nounwind readnone
declare i32 @llvm.arm.qadd16(i32, i32) nounwind
declare i32 @llvm.arm.qasx(i32, i32) nounwind
declare i32 @llvm.arm.qsax(i32, i32) nounwind
declare i32 @llvm.arm.qsub16(i32, i32) nounwind
declare i32 @llvm.arm.sadd16(i32, i32) nounwind
declare i32 @llvm.arm.sasx(i32, i32) nounwind
declare i32 @llvm.arm.shadd16(i32, i32) nounwind
declare i32 @llvm.arm.shasx(i32, i32) nounwind
declare i32 @llvm.arm.shsax(i32, i32) nounwind
declare i32 @llvm.arm.shsub16(i32, i32) nounwind
declare i32 @llvm.arm.ssax(i32, i32) nounwind
declare i32 @llvm.arm.ssub16(i32, i32) nounwind
declare i32 @llvm.arm.uadd16(i32, i32) nounwind
declare i32 @llvm.arm.uasx(i32, i32) nounwind
declare i32 @llvm.arm.usax(i32, i32) nounwind
declare i32 @llvm.arm.uhadd16(i32, i32) nounwind
declare i32 @llvm.arm.uhasx(i32, i32) nounwind
declare i32 @llvm.arm.uhsax(i32, i32) nounwind
declare i32 @llvm.arm.uhsub16(i32, i32) nounwind
declare i32 @llvm.arm.uqadd16(i32, i32) nounwind
declare i32 @llvm.arm.uqasx(i32, i32) nounwind
declare i32 @llvm.arm.uqsax(i32, i32) nounwind
declare i32 @llvm.arm.uqsub16(i32, i32) nounwind
declare i32 @llvm.arm.usub16(i32, i32) nounwind
declare i32 @llvm.arm.smlad(i32, i32, i32) nounwind
declare i32 @llvm.arm.smladx(i32, i32, i32) nounwind
declare i64 @llvm.arm.smlald(i32, i32, i64) nounwind
declare i64 @llvm.arm.smlaldx(i32, i32, i64) nounwind
declare i32 @llvm.arm.smlsd(i32, i32, i32) nounwind
declare i32 @llvm.arm.smlsdx(i32, i32, i32) nounwind
declare i64 @llvm.arm.smlsld(i32, i32, i64) nounwind
declare i64 @llvm.arm.smlsldx(i32, i32, i64) nounwind
declare i32 @llvm.arm.smuad(i32, i32) nounwind
declare i32 @llvm.arm.smuadx(i32, i32) nounwind
declare i32 @llvm.arm.smusd(i32, i32) nounwind
declare i32 @llvm.arm.smusdx(i32, i32) nounwind
