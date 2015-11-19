; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit integer operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i64 @llvm.ctlz.i64(i64, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i64 @llvm.ctpop.i64(i64)

; CHECK-LABEL: add64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.add $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add64(i64 %x, i64 %y) {
  %a = add i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: sub64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.sub $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub64(i64 %x, i64 %y) {
  %a = sub i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: mul64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.mul $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @mul64(i64 %x, i64 %y) {
  %a = mul i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: sdiv64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.div_s $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sdiv64(i64 %x, i64 %y) {
  %a = sdiv i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: udiv64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.div_u $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @udiv64(i64 %x, i64 %y) {
  %a = udiv i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: srem64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.rem_s $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @srem64(i64 %x, i64 %y) {
  %a = srem i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: urem64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.rem_u $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @urem64(i64 %x, i64 %y) {
  %a = urem i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: and64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.and $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and64(i64 %x, i64 %y) {
  %a = and i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: or64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.or $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or64(i64 %x, i64 %y) {
  %a = or i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: xor64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.xor $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor64(i64 %x, i64 %y) {
  %a = xor i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: shl64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.shl $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @shl64(i64 %x, i64 %y) {
  %a = shl i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: shr64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.shr_u $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @shr64(i64 %x, i64 %y) {
  %a = lshr i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: sar64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.shr_s $push0, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sar64(i64 %x, i64 %y) {
  %a = ashr i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: clz64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.clz $push0, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @clz64(i64 %x) {
  %a = call i64 @llvm.ctlz.i64(i64 %x, i1 false)
  ret i64 %a
}

; CHECK-LABEL: clz64_zero_undef:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.clz $push0, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @clz64_zero_undef(i64 %x) {
  %a = call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  ret i64 %a
}

; CHECK-LABEL: ctz64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.ctz $push0, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @ctz64(i64 %x) {
  %a = call i64 @llvm.cttz.i64(i64 %x, i1 false)
  ret i64 %a
}

; CHECK-LABEL: ctz64_zero_undef:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.ctz $push0, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @ctz64_zero_undef(i64 %x) {
  %a = call i64 @llvm.cttz.i64(i64 %x, i1 true)
  ret i64 %a
}

; CHECK-LABEL: popcnt64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.popcnt $push0, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @popcnt64(i64 %x) {
  %a = call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %a
}
