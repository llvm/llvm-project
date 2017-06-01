; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit integer operations assemble as expected.

target datalayout = "E-m:e-p:32:32-i64:64-a:0:32-n32-S64"
target triple = "lanai"

; Function Attrs: nounwind readnone
declare i32 @llvm.ctpop.i32(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.cttz.i32(i32, i1) #1

; CHECK-LABEL: add32:
; CHECK: add  %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @add32(i32 %x, i32 %y) {
  %a = add i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: sub32:
; CHECK: sub  %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @sub32(i32 %x, i32 %y) {
  %a = sub i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: mul32:
; CHECK: bt __mulsi3
define i32 @mul32(i32 %x, i32 %y) {
  %a = mul i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: sdiv32:
; CHECK: bt __divsi3
define i32 @sdiv32(i32 %x, i32 %y) {
  %a = sdiv i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: udiv32:
; CHECK: bt __udivsi3
define i32 @udiv32(i32 %x, i32 %y) {
  %a = udiv i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: srem32:
; CHECK: bt __modsi3
define i32 @srem32(i32 %x, i32 %y) {
  %a = srem i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: urem32:
; CHECK: bt __umodsi3
define i32 @urem32(i32 %x, i32 %y) {
  %a = urem i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: and32:
; CHECK: and %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @and32(i32 %x, i32 %y) {
  %a = and i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: or32:
; CHECK: or %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @or32(i32 %x, i32 %y) {
  %a = or i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: xor32:
; CHECK: xor %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @xor32(i32 %x, i32 %y) {
  %a = xor i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: shl32:
; CHECK: sh %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @shl32(i32 %x, i32 %y) {
  %a = shl i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: shr32:
; CHECK: sub %r0, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: sh %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @shr32(i32 %x, i32 %y) {
  %a = lshr i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: sar32
; CHECK: sub %r0, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: sha %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @sar32(i32 %x, i32 %y) {
  %a = ashr i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: clz32:
; CHECK: leadz %r{{[0-9]+}}, %rv
define i32 @clz32(i32 %x) {
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: clz32_zero_undef:
; CHECK-NOT: sub.f
; CHECK: leadz %r{{[0-9]+}}, %rv
define i32 @clz32_zero_undef(i32 %x) {
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  ret i32 %a
}

; CHECK-LABEL: ctz32:
; CHECK: trailz %r{{[0-9]+}}, %rv
define i32 @ctz32(i32 %x) {
  %a = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: ctz32_zero_undef:
; CHECK-NOT: sub.f
; CHECK: trailz  %r{{[0-9]+}}, %rv
define i32 @ctz32_zero_undef(i32 %x) {
  %a = call i32 @llvm.cttz.i32(i32 %x, i1 true)
  ret i32 %a
}

; CHECK-LABEL: popcnt32:
; CHECK: popc %r{{[0-9]+}}, %rv
define i32 @popcnt32(i32 %x) {
  %a = call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %a
}
