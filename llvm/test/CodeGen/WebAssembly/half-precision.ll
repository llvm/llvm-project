; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+half-precision,+simd128 | FileCheck %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+half-precision,+simd128 | FileCheck %s

declare float @llvm.wasm.loadf32.f16(ptr)
declare void @llvm.wasm.storef16.f32(float, ptr)

; CHECK-LABEL: ldf16_32:
; CHECK:      f32.load_f16 $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM0]]{{$}}
define float @ldf16_32(ptr %p) {
  %v = call float @llvm.wasm.loadf16.f32(ptr %p)
  ret float %v
}

; CHECK-LABEL: stf16_32:
; CHECK:       f32.store_f16 0($1), $0
; CHECK-NEXT:  return
define void @stf16_32(float %v, ptr %p) {
  tail call void @llvm.wasm.storef16.f32(float %v, ptr %p)
  ret void
}

; CHECK-LABEL: splat_v8f16:
; CHECK:       f16x8.splat $push0=, $0
; CHECK-NEXT:  return $pop0
define <8 x half> @splat_v8f16(float %x) {
  %v = call <8 x half> @llvm.wasm.splat.f16x8(float %x)
  ret <8 x half> %v
}

; CHECK-LABEL: extract_lane_v8f16:
; CHECK:       f16x8.extract_lane $push0=, $0, 1
; CHECK-NEXT:  return $pop0
define float @extract_lane_v8f16(<8 x half> %v) {
  %r = call float @llvm.wasm.extract.lane.f16x8(<8 x half> %v, i32 1)
  ret float %r
}

; CHECK-LABEL: add_v8f16:
; CHECK:       f16x8.add $push0=, $0, $1
; CHECK-NEXT:  return $pop0
define <8 x half> @add_v8f16(<8 x half> %a, <8 x half> %b) {
  %r = fadd <8 x half> %a, %b
  ret <8 x half> %r
}

; CHECK-LABEL: sub_v8f16:
; CHECK:       f16x8.sub $push0=, $0, $1
; CHECK-NEXT:  return $pop0
define <8 x half> @sub_v8f16(<8 x half> %a, <8 x half> %b) {
  %r = fsub <8 x half> %a, %b
  ret <8 x half> %r
}

; CHECK-LABEL: mul_v8f16:
; CHECK:       f16x8.mul $push0=, $0, $1
; CHECK-NEXT:  return $pop0
define <8 x half> @mul_v8f16(<8 x half> %a, <8 x half> %b) {
  %r = fmul <8 x half> %a, %b
  ret <8 x half> %r
}

; CHECK-LABEL: div_v8f16:
; CHECK:       f16x8.div $push0=, $0, $1
; CHECK-NEXT:  return $pop0
define <8 x half> @div_v8f16(<8 x half> %a, <8 x half> %b) {
  %r = fdiv <8 x half> %a, %b
  ret <8 x half> %r
}

; CHECK-LABEL: min_intrinsic_v8f16:
; CHECK:       f16x8.min $push0=, $0, $1
; CHECK-NEXT:  return $pop0
declare <8 x half> @llvm.minimum.v8f16(<8 x half>, <8 x half>)
define <8 x half> @min_intrinsic_v8f16(<8 x half> %x, <8 x half> %y) {
  %a = call <8 x half> @llvm.minimum.v8f16(<8 x half> %x, <8 x half> %y)
  ret <8 x half> %a
}

; CHECK-LABEL: max_intrinsic_v8f16:
; CHECK:       f16x8.max $push0=, $0, $1
; CHECK-NEXT:  return $pop0
declare <8 x half> @llvm.maximum.v8f16(<8 x half>, <8 x half>)
define <8 x half> @max_intrinsic_v8f16(<8 x half> %x, <8 x half> %y) {
  %a = call <8 x half> @llvm.maximum.v8f16(<8 x half> %x, <8 x half> %y)
  ret <8 x half> %a
}

; CHECK-LABEL: pmin_intrinsic_v8f16:
; CHECK:       f16x8.pmin $push0=, $0, $1
; CHECK-NEXT:  return $pop0
declare <8 x half> @llvm.wasm.pmin.v8f16(<8 x half>, <8 x half>)
define <8 x half> @pmin_intrinsic_v8f16(<8 x half> %a, <8 x half> %b) {
  %v = call <8 x half> @llvm.wasm.pmin.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %v
}

; CHECK-LABEL: pmax_intrinsic_v8f16:
; CHECK:       f16x8.pmax $push0=, $0, $1
; CHECK-NEXT:  return $pop0
declare <8 x half> @llvm.wasm.pmax.v8f16(<8 x half>, <8 x half>)
define <8 x half> @pmax_intrinsic_v8f16(<8 x half> %a, <8 x half> %b) {
  %v = call <8 x half> @llvm.wasm.pmax.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %v
}
