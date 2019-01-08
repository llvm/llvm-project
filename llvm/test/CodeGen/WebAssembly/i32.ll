; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Test that basic 32-bit integer operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i32 @llvm.ctpop.i32(i32)

; CHECK-LABEL: add32:
; CHECK-NEXT: .functype add32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.add $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add32(i32 %x, i32 %y) {
  %a = add i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: sub32:
; CHECK-NEXT: .functype sub32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.sub $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sub32(i32 %x, i32 %y) {
  %a = sub i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: mul32:
; CHECK-NEXT: .functype mul32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.mul $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @mul32(i32 %x, i32 %y) {
  %a = mul i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: sdiv32:
; CHECK-NEXT: .functype sdiv32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.div_s $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sdiv32(i32 %x, i32 %y) {
  %a = sdiv i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: udiv32:
; CHECK-NEXT: .functype udiv32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.div_u $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @udiv32(i32 %x, i32 %y) {
  %a = udiv i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: srem32:
; CHECK-NEXT: .functype srem32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.rem_s $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @srem32(i32 %x, i32 %y) {
  %a = srem i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: urem32:
; CHECK-NEXT: .functype urem32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.rem_u $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @urem32(i32 %x, i32 %y) {
  %a = urem i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: and32:
; CHECK-NEXT: .functype and32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.and $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @and32(i32 %x, i32 %y) {
  %a = and i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: or32:
; CHECK-NEXT: .functype or32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.or $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @or32(i32 %x, i32 %y) {
  %a = or i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: xor32:
; CHECK-NEXT: .functype xor32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.xor $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xor32(i32 %x, i32 %y) {
  %a = xor i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: shl32:
; CHECK-NEXT: .functype shl32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.shl $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @shl32(i32 %x, i32 %y) {
  %a = shl i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: shr32:
; CHECK-NEXT: .functype shr32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.shr_u $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @shr32(i32 %x, i32 %y) {
  %a = lshr i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: sar32:
; CHECK-NEXT: .functype sar32 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.shr_s $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sar32(i32 %x, i32 %y) {
  %a = ashr i32 %x, %y
  ret i32 %a
}

; CHECK-LABEL: clz32:
; CHECK-NEXT: .functype clz32 (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.clz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @clz32(i32 %x) {
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: clz32_zero_undef:
; CHECK-NEXT: .functype clz32_zero_undef (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.clz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @clz32_zero_undef(i32 %x) {
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  ret i32 %a
}

; CHECK-LABEL: ctz32:
; CHECK-NEXT: .functype ctz32 (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.ctz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @ctz32(i32 %x) {
  %a = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: ctz32_zero_undef:
; CHECK-NEXT: .functype ctz32_zero_undef (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.ctz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @ctz32_zero_undef(i32 %x) {
  %a = call i32 @llvm.cttz.i32(i32 %x, i1 true)
  ret i32 %a
}

; CHECK-LABEL: popcnt32:
; CHECK-NEXT: .functype popcnt32 (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.popcnt $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @popcnt32(i32 %x) {
  %a = call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %a
}

; CHECK-LABEL: eqz32:
; CHECK-NEXT: .functype eqz32 (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.eqz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @eqz32(i32 %x) {
  %a = icmp eq i32 %x, 0
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: rotl:
; CHECK-NEXT: .functype rotl (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.rotl $push0=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: return $pop0{{$}}
define i32 @rotl(i32 %x, i32 %y) {
  %z = sub i32 32, %y
  %b = shl i32 %x, %y
  %c = lshr i32 %x, %z
  %d = or i32 %b, %c
  ret i32 %d
}

; CHECK-LABEL: masked_rotl:
; CHECK-NEXT: .functype masked_rotl (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.rotl $push0=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: return $pop0{{$}}
define i32 @masked_rotl(i32 %x, i32 %y) {
  %a = and i32 %y, 31
  %z = sub i32 32, %a
  %b = shl i32 %x, %a
  %c = lshr i32 %x, %z
  %d = or i32 %b, %c
  ret i32 %d
}

; CHECK-LABEL: rotr:
; CHECK-NEXT: .functype rotr (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.rotr $push0=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: return $pop0{{$}}
define i32 @rotr(i32 %x, i32 %y) {
  %z = sub i32 32, %y
  %b = lshr i32 %x, %y
  %c = shl i32 %x, %z
  %d = or i32 %b, %c
  ret i32 %d
}

; CHECK-LABEL: masked_rotr:
; CHECK-NEXT: .functype masked_rotr (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.rotr $push0=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: return $pop0{{$}}
define i32 @masked_rotr(i32 %x, i32 %y) {
  %a = and i32 %y, 31
  %z = sub i32 32, %a
  %b = lshr i32 %x, %a
  %c = shl i32 %x, %z
  %d = or i32 %b, %c
  ret i32 %d
}
