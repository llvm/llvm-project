; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define signext i32 @foo(i32 signext %a) #0 {
entry:
  %mul = mul nsw i32 %a, %a
  %shr2 = lshr i32 %mul, 5
  ret i32 %shr2

; CHECK-LABEL: @foo
; CHECK-NOT: rldicl 3, {{[0-9]+}}, 0, 32
; CHECK: blr
}

define zeroext i32 @test6(i32 zeroext %x) #0 {
entry:
  %and = lshr i32 %x, 16
  %shr = and i32 %and, 255
  %and1 = shl i32 %x, 16
  %shl = and i32 %and1, 16711680
  %or = or i32 %shr, %shl
  ret i32 %or

; CHECK-LABEL: @test6
; CHECK-NOT: rldicl 3, {{[0-9]+}}, 0, 32
; CHECK: blr
}

define zeroext i32 @min(i32 zeroext %a, i32 zeroext %b) #0 {
entry:
  %cmp = icmp ule i32 %a, %b
  %cond = select i1 %cmp, i32 %a, i32 %b
  ret i32 %cond

; CHECK-LABEL: @min
; CHECK-NOT: rldicl 3, {{[0-9]+}}, 0, 32
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare i32 @llvm.bswap.i32(i32) #0

; Function Attrs: nounwind readonly
define zeroext i32 @bs32(i32* nocapture readonly %x) #1 {
entry:
  %0 = load i32, i32* %x, align 4
  %1 = tail call i32 @llvm.bswap.i32(i32 %0)
  ret i32 %1

; CHECK-LABEL: @bs32
; CHECK-NOT: rldicl 3, {{[0-9]+}}, 0, 32
; CHECK: blr
}

; Function Attrs: nounwind readonly
define zeroext i16 @bs16(i16* nocapture readonly %x) #1 {
entry:
  %0 = load i16, i16* %x, align 2
  %1 = tail call i16 @llvm.bswap.i16(i16 %0)
  ret i16 %1

; CHECK-LABEL: @bs16
; CHECK-NOT: rldicl 3, {{[0-9]+}}, 0, 32
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare i16 @llvm.bswap.i16(i16) #0

; Function Attrs: nounwind readnone
define zeroext i32 @ctlz32(i32 zeroext %x) #0 {
entry:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %0

; CHECK-LABEL: @ctlz32
; CHECK-NOT: rldicl 3, {{[0-9]+}}, 0, 32
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) #0


attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly }

