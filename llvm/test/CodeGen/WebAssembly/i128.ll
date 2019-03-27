; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Test that basic 128-bit integer operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i128 @llvm.ctlz.i128(i128, i1)
declare i128 @llvm.cttz.i128(i128, i1)
declare i128 @llvm.ctpop.i128(i128)

; CHECK-LABEL: add128:
; CHECK-NEXT: .functype add128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK-NOT:  .result
; CHECK:      i64.add
; CHECK:      i64.store
; CHECK:      i64.add
; CHECK:      i64.store
; CHECK-NEXT: return{{$}}
define i128 @add128(i128 %x, i128 %y) {
  %a = add i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: sub128:
; CHECK-NEXT: .functype sub128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK:      i64.sub
; CHECK:      i64.store
; CHECK:      i64.sub
; CHECK:      i64.store
; CHECK-NEXT: return{{$}}
define i128 @sub128(i128 %x, i128 %y) {
  %a = sub i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: mul128:
; CHECK-NEXT: .functype mul128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __multi3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @mul128(i128 %x, i128 %y) {
  %a = mul i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: sdiv128:
; CHECK-NEXT: .functype sdiv128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __divti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @sdiv128(i128 %x, i128 %y) {
  %a = sdiv i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: udiv128:
; CHECK-NEXT: .functype udiv128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __udivti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @udiv128(i128 %x, i128 %y) {
  %a = udiv i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: srem128:
; CHECK-NEXT: .functype srem128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __modti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @srem128(i128 %x, i128 %y) {
  %a = srem i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: urem128:
; CHECK-NEXT: .functype urem128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __umodti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @urem128(i128 %x, i128 %y) {
  %a = urem i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: and128:
; CHECK-NEXT: .functype and128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK-NOT:  .result
; CHECK:      i64.and
; CHECK:      i64.store
; CHECK:      i64.and
; CHECK:      i64.store
; CHECK-NEXT: return{{$}}
define i128 @and128(i128 %x, i128 %y) {
  %a = and i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: or128:
; CHECK-NEXT: .functype or128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK:      i64.or
; CHECK:      i64.store
; CHECK:      i64.or
; CHECK:      i64.store
; CHECK-NEXT: return{{$}}
define i128 @or128(i128 %x, i128 %y) {
  %a = or i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: xor128:
; CHECK-NEXT: .functype xor128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK:      i64.xor
; CHECK:      i64.store
; CHECK:      i64.xor
; CHECK:      i64.store
; CHECK-NEXT: return{{$}}
define i128 @xor128(i128 %x, i128 %y) {
  %a = xor i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: shl128:
; CHECK-NEXT: .functype shl128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __ashlti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @shl128(i128 %x, i128 %y) {
  %a = shl i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: shr128:
; CHECK-NEXT: .functype shr128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __lshrti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @shr128(i128 %x, i128 %y) {
  %a = lshr i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: sar128:
; CHECK-NEXT: .functype sar128 (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __ashrti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @sar128(i128 %x, i128 %y) {
  %a = ashr i128 %x, %y
  ret i128 %a
}

; CHECK-LABEL: clz128:
; CHECK-NEXT: .functype clz128 (i32, i64, i64) -> (){{$}}
; CHECK-NOT:  .result
; CHECK:      i64.clz
; CHECK:      i64.clz
; CHECK:      return{{$}}
define i128 @clz128(i128 %x) {
  %a = call i128 @llvm.ctlz.i128(i128 %x, i1 false)
  ret i128 %a
}

; CHECK-LABEL: clz128_zero_undef:
; CHECK-NEXT: .functype clz128_zero_undef (i32, i64, i64) -> (){{$}}
; CHECK:      i64.clz
; CHECK:      i64.clz
; CHECK:      return{{$}}
define i128 @clz128_zero_undef(i128 %x) {
  %a = call i128 @llvm.ctlz.i128(i128 %x, i1 true)
  ret i128 %a
}

; CHECK-LABEL: ctz128:
; CHECK-NEXT: .functype ctz128 (i32, i64, i64) -> (){{$}}
; CHECK:      i64.ctz
; CHECK:      i64.ctz
; CHECK:      return{{$}}
define i128 @ctz128(i128 %x) {
  %a = call i128 @llvm.cttz.i128(i128 %x, i1 false)
  ret i128 %a
}

; CHECK-LABEL: ctz128_zero_undef:
; CHECK-NEXT: .functype ctz128_zero_undef (i32, i64, i64) -> (){{$}}
; CHECK:      i64.ctz
; CHECK:      i64.ctz
; CHECK:      return{{$}}
define i128 @ctz128_zero_undef(i128 %x) {
  %a = call i128 @llvm.cttz.i128(i128 %x, i1 true)
  ret i128 %a
}

; CHECK-LABEL: popcnt128:
; CHECK-NEXT: .functype popcnt128 (i32, i64, i64) -> (){{$}}
; CHECK:      i64.popcnt
; CHECK:      i64.popcnt
; CHECK:      return{{$}}
define i128 @popcnt128(i128 %x) {
  %a = call i128 @llvm.ctpop.i128(i128 %x)
  ret i128 %a
}

; CHECK-LABEL: eqz128:
; CHECK-NEXT: .functype eqz128 (i64, i64) -> (i32){{$}}
; CHECK:     i64.or
; CHECK:     i64.eqz
; CHECK:     return $
define i32 @eqz128(i128 %x) {
  %a = icmp eq i128 %x, 0
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: rotl:
; CHECK-NEXT: .functype rotl (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __ashlti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: call __lshrti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @rotl(i128 %x, i128 %y) {
  %z = sub i128 128, %y
  %b = shl i128 %x, %y
  %c = lshr i128 %x, %z
  %d = or i128 %b, %c
  ret i128 %d
}

; CHECK-LABEL: masked_rotl:
; CHECK-NEXT: .functype masked_rotl (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __ashlti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: call __lshrti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @masked_rotl(i128 %x, i128 %y) {
  %a = and i128 %y, 127
  %z = sub i128 128, %a
  %b = shl i128 %x, %a
  %c = lshr i128 %x, %z
  %d = or i128 %b, %c
  ret i128 %d
}

; CHECK-LABEL: rotr:
; CHECK-NEXT: .functype rotr (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __lshrti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: call __ashlti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @rotr(i128 %x, i128 %y) {
  %z = sub i128 128, %y
  %b = lshr i128 %x, %y
  %c = shl i128 %x, %z
  %d = or i128 %b, %c
  ret i128 %d
}

; CHECK-LABEL: masked_rotr:
; CHECK-NEXT: .functype masked_rotr (i32, i64, i64, i64, i64) -> (){{$}}
; CHECK: call __lshrti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: call __ashlti3, ${{.+}}, ${{.+}}, ${{.+}}, ${{.+}}{{$}}
; CHECK: return{{$}}
define i128 @masked_rotr(i128 %x, i128 %y) {
  %a = and i128 %y, 127
  %z = sub i128 128, %a
  %b = lshr i128 %x, %a
  %c = shl i128 %x, %z
  %d = or i128 %b, %c
  ret i128 %d
}
