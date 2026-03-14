; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Test that basic 64-bit integer operations assemble as expected.

target triple = "wasm32-unknown-unknown"

declare i64 @llvm.ctlz.i64(i64, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i64 @llvm.ctpop.i64(i64)

; CHECK-LABEL: add64:
; CHECK-NEXT: .functype add64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.add $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add64(i64 %x, i64 %y) {
  %a = add i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: sub64:
; CHECK-NEXT: .functype sub64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.sub $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub64(i64 %x, i64 %y) {
  %a = sub i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: mul64:
; CHECK-NEXT: .functype mul64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.mul $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @mul64(i64 %x, i64 %y) {
  %a = mul i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: sdiv64:
; CHECK-NEXT: .functype sdiv64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.div_s $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sdiv64(i64 %x, i64 %y) {
  %a = sdiv i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: udiv64:
; CHECK-NEXT: .functype udiv64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.div_u $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @udiv64(i64 %x, i64 %y) {
  %a = udiv i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: srem64:
; CHECK-NEXT: .functype srem64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.rem_s $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @srem64(i64 %x, i64 %y) {
  %a = srem i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: urem64:
; CHECK-NEXT: .functype urem64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.rem_u $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @urem64(i64 %x, i64 %y) {
  %a = urem i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: and64:
; CHECK-NEXT: .functype and64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.and $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and64(i64 %x, i64 %y) {
  %a = and i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: or64:
; CHECK-NEXT: .functype or64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.or $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or64(i64 %x, i64 %y) {
  %a = or i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: xor64:
; CHECK-NEXT: .functype xor64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.xor $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor64(i64 %x, i64 %y) {
  %a = xor i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: shl64:
; CHECK-NEXT: .functype shl64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.shl $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @shl64(i64 %x, i64 %y) {
  %a = shl i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: shr64:
; CHECK-NEXT: .functype shr64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.shr_u $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @shr64(i64 %x, i64 %y) {
  %a = lshr i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: sar64:
; CHECK-NEXT: .functype sar64 (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.shr_s $push0=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sar64(i64 %x, i64 %y) {
  %a = ashr i64 %x, %y
  ret i64 %a
}

; CHECK-LABEL: clz64:
; CHECK-NEXT: .functype clz64 (i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.clz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @clz64(i64 %x) {
  %a = call i64 @llvm.ctlz.i64(i64 %x, i1 false)
  ret i64 %a
}

; CHECK-LABEL: clz64_zero_undef:
; CHECK-NEXT: .functype clz64_zero_undef (i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.clz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @clz64_zero_undef(i64 %x) {
  %a = call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  ret i64 %a
}

; CHECK-LABEL: ctz64:
; CHECK-NEXT: .functype ctz64 (i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.ctz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @ctz64(i64 %x) {
  %a = call i64 @llvm.cttz.i64(i64 %x, i1 false)
  ret i64 %a
}

; CHECK-LABEL: ctz64_zero_undef:
; CHECK-NEXT: .functype ctz64_zero_undef (i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.ctz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @ctz64_zero_undef(i64 %x) {
  %a = call i64 @llvm.cttz.i64(i64 %x, i1 true)
  ret i64 %a
}

; CHECK-LABEL: popcnt64:
; CHECK-NEXT: .functype popcnt64 (i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.popcnt $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @popcnt64(i64 %x) {
  %a = call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %a
}

; CHECK-LABEL: eqz64:
; CHECK-NEXT: .functype eqz64 (i64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.eqz $push0=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @eqz64(i64 %x) {
  %a = icmp eq i64 %x, 0
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: rotl:
; CHECK-NEXT: .functype rotl (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.rotl $push0=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: return $pop0{{$}}
define i64 @rotl(i64 %x, i64 %y) {
  %z = sub i64 64, %y
  %b = shl i64 %x, %y
  %c = lshr i64 %x, %z
  %d = or i64 %b, %c
  ret i64 %d
}

; CHECK-LABEL: masked_rotl:
; CHECK-NEXT: .functype masked_rotl (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.rotl $push0=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: return $pop0{{$}}
define i64 @masked_rotl(i64 %x, i64 %y) {
  %a = and i64 %y, 63
  %z = sub i64 64, %a
  %b = shl i64 %x, %a
  %c = lshr i64 %x, %z
  %d = or i64 %b, %c
  ret i64 %d
}

; CHECK-LABEL: rotr:
; CHECK-NEXT: .functype rotr (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.rotr $push0=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: return $pop0{{$}}
define i64 @rotr(i64 %x, i64 %y) {
  %z = sub i64 64, %y
  %b = lshr i64 %x, %y
  %c = shl i64 %x, %z
  %d = or i64 %b, %c
  ret i64 %d
}

; CHECK-LABEL: masked_rotr:
; CHECK-NEXT: .functype masked_rotr (i64, i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.rotr $push0=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: return $pop0{{$}}
define i64 @masked_rotr(i64 %x, i64 %y) {
  %a = and i64 %y, 63
  %z = sub i64 64, %a
  %b = lshr i64 %x, %a
  %c = shl i64 %x, %z
  %d = or i64 %b, %c
  ret i64 %d
}
