; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+fp16,+simd128 | FileCheck %s

; Tests for `llvm.wasm.*.*f16` intrinsics

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

; CHECK-LABEL: const_splat_v8f16:
; CHECK:       v128.const      $push0=, 20800, 0, 0, 0, 0, 0, 0, 20800
; CHECK-NEXT:  return $pop0
define <8 x half> @const_splat_v8f16() {
  ret <8 x half> <half 42., half 0., half 0., half 0., half 0., half 0., half 0., half 42.>
}

; CHECK-LABEL: extract_lane_v8f16:
; CHECK:       f16x8.extract_lane $push0=, $0, 1
; CHECK-NEXT:  return $pop0
define float @extract_lane_v8f16(<8 x half> %v) {
  %r = call float @llvm.wasm.extract.lane.f16x8(<8 x half> %v, i32 1)
  ret float %r
}

; CHECK-LABEL: replace_lane_v8f16:
; CHECK:       f16x8.replace_lane $push0=, $0, 1, $1
; CHECK-NEXT:  return $pop0
define <8 x half> @replace_lane_v8f16(<8 x half> %v, float %f) {
  %r = call <8 x half> @llvm.wasm.replace.lane.f16x8(<8 x half> %v, i32 1, float %f)
  ret <8 x half> %r
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

; CHECK-LABEL: compare_oeq_v8f16:
; CHECK-NEXT: .functype compare_oeq_v8f16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f16x8.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_oeq_v8f16 (<8 x half> %x, <8 x half> %y) {
  %res = fcmp oeq <8 x half> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_une_v8f16:
; CHECK-NEXT: .functype compare_une_v8f16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f16x8.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_une_v8f16 (<8 x half> %x, <8 x half> %y) {
  %res = fcmp une <8 x half> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_olt_v8f16:
; CHECK-NEXT: .functype compare_olt_v8f16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f16x8.lt $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_olt_v8f16 (<8 x half> %x, <8 x half> %y) {
  %res = fcmp olt <8 x half> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_ogt_v8f16:
; CHECK-NEXT: .functype compare_ogt_v8f16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f16x8.gt $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_ogt_v8f16 (<8 x half> %x, <8 x half> %y) {
  %res = fcmp ogt <8 x half> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_ole_v8f16:
; CHECK-NEXT: .functype compare_ole_v8f16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f16x8.le $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_ole_v8f16 (<8 x half> %x, <8 x half> %y) {
  %res = fcmp ole <8 x half> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_oge_v8f16:
; CHECK-NEXT: .functype compare_oge_v8f16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f16x8.ge $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_oge_v8f16 (<8 x half> %x, <8 x half> %y) {
  %res = fcmp oge <8 x half> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_ule_v8f16:
; CHECK-NEXT: .functype compare_ule_v8f16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f16x8.gt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_ule_v8f16 (<8 x half> %x, <8 x half> %y) {
  %res = fcmp ule <8 x half> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: abs_v8f16:
; CHECK-NEXT:  .functype abs_v8f16 (v128) -> (v128)
; CHECK-NEXT:  f16x8.abs $push0=, $0
; CHECK-NEXT:  return $pop0
declare <8 x half> @llvm.fabs.v8f16(<8 x half>) nounwind readnone
define <8 x half> @abs_v8f16(<8 x half> %x) {
  %a = call <8 x half> @llvm.fabs.v8f16(<8 x half> %x)
  ret <8 x half> %a
}

; CHECK-LABEL: neg_v8f16:
; CHECK-NEXT:  .functype neg_v8f16 (v128) -> (v128)
; CHECK-NEXT:  f16x8.neg $push0=, $0
; CHECK-NEXT:  return $pop0
define <8 x half> @neg_v8f16(<8 x half> %x) {
  %a = fsub nsz <8 x half> <half 0., half 0., half 0., half 0., half 0., half 0., half 0., half 0.>, %x
  ret <8 x half> %a
}

; CHECK-LABEL: sqrt_v8f16:
; CHECK-NEXT:  .functype sqrt_v8f16 (v128) -> (v128)
; CHECK-NEXT:  f16x8.sqrt $push0=, $0
; CHECK-NEXT:  return $pop0
declare <8 x half> @llvm.sqrt.v8f16(<8 x half> %x)
define <8 x half> @sqrt_v8f16(<8 x half> %x) {
  %a = call <8 x half> @llvm.sqrt.v8f16(<8 x half> %x)
  ret <8 x half> %a
}

; CHECK-LABEL: ceil_v8f16:
; CHECK-NEXT:  .functype ceil_v8f16 (v128) -> (v128){{$}}
; CHECK-NEXT:  f16x8.ceil $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT:  return $pop[[R]]{{$}}
declare <8 x half> @llvm.ceil.v8f16(<8 x half>)
define <8 x half> @ceil_v8f16(<8 x half> %a) {
  %v = call <8 x half> @llvm.ceil.v8f16(<8 x half> %a)
  ret <8 x half> %v
}

; CHECK-LABEL: floor_v8f16:
; CHECK-NEXT:  .functype floor_v8f16 (v128) -> (v128){{$}}
; CHECK-NEXT:  f16x8.floor $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT:  return $pop[[R]]{{$}}
declare <8 x half> @llvm.floor.v8f16(<8 x half>)
define <8 x half> @floor_v8f16(<8 x half> %a) {
  %v = call <8 x half> @llvm.floor.v8f16(<8 x half> %a)
  ret <8 x half> %v
}

; CHECK-LABEL: trunc_v8f16:
; CHECK-NEXT:  .functype trunc_v8f16 (v128) -> (v128){{$}}
; CHECK-NEXT:  f16x8.trunc $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT:  return $pop[[R]]{{$}}
declare <8 x half> @llvm.trunc.v8f16(<8 x half>)
define <8 x half> @trunc_v8f16(<8 x half> %a) {
  %v = call <8 x half> @llvm.trunc.v8f16(<8 x half> %a)
  ret <8 x half> %v
}

; CHECK-LABEL: nearest_v8f16:
; CHECK-NEXT:  .functype nearest_v8f16 (v128) -> (v128){{$}}
; CHECK-NEXT:  f16x8.nearest $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT:  return $pop[[R]]{{$}}
declare <8 x half> @llvm.nearbyint.v8f16(<8 x half>)
define <8 x half> @nearest_v8f16(<8 x half> %a) {
  %v = call <8 x half> @llvm.nearbyint.v8f16(<8 x half> %a)
  ret <8 x half> %v
}

; CHECK-LABEL: nearest_v8f16_via_rint:
; CHECK-NEXT:  .functype nearest_v8f16_via_rint (v128) -> (v128){{$}}
; CHECK-NEXT:  f16x8.nearest $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT:  return $pop[[R]]{{$}}
declare <8 x half> @llvm.rint.v8f16(<8 x half>)
define <8 x half> @nearest_v8f16_via_rint(<8 x half> %a) {
  %v = call <8 x half> @llvm.rint.v8f16(<8 x half> %a)
  ret <8 x half> %v
}

; CHECK-LABEL: nearest_v8f16_via_roundeven:
; CHECK-NEXT:  .functype nearest_v8f16_via_roundeven (v128) -> (v128){{$}}
; CHECK-NEXT:  f16x8.nearest $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT:  return $pop[[R]]{{$}}
declare <8 x half> @llvm.roundeven.v8f16(<8 x half>)
define <8 x half> @nearest_v8f16_via_roundeven(<8 x half> %a) {
  %v = call <8 x half> @llvm.roundeven.v8f16(<8 x half> %a)
  ret <8 x half> %v
}

define <8 x half> @convert_s_v8f16(<8 x i16> %x) {
; CHECK-LABEL: convert_s_v8f16:
; CHECK:         .functype convert_s_v8f16 (v128) -> (v128)
; CHECK-NEXT:    f16x8.convert_i16x8_s $push0=, $0
; CHECK-NEXT:    return $pop[[R]]{{$}}
  %a = sitofp <8 x i16> %x to <8 x half>
  ret <8 x half> %a
}

define <8 x half> @convert_u_v8f16(<8 x i16> %x) {
; CHECK-LABEL: convert_u_v8f16:
; CHECK:         .functype convert_u_v8f16 (v128) -> (v128)
; CHECK-NEXT:    f16x8.convert_i16x8_u $push0=, $0
; CHECK-NEXT:    return $pop[[R]]{{$}}
  %a = uitofp <8 x i16> %x to <8 x half>
  ret <8 x half> %a
}

define <8 x i16> @trunc_sat_s_v8i16(<8 x half> %x) {
; CHECK-LABEL: trunc_sat_s_v8i16:
; CHECK:         .functype trunc_sat_s_v8i16 (v128) -> (v128)
; CHECK-NEXT:    i16x8.trunc_sat_f16x8_s $push0=, $0
; CHECK-NEXT:    return $pop[[R]]{{$}}
  %a = fptosi <8 x half> %x to <8 x i16>
  ret <8 x i16> %a
}

define <8 x i16> @trunc_sat_u_v8i16(<8 x half> %x) {
; CHECK-LABEL: trunc_sat_u_v8i16:
; CHECK:         .functype trunc_sat_u_v8i16 (v128) -> (v128)
; CHECK-NEXT:    i16x8.trunc_sat_f16x8_u $push0=, $0
; CHECK-NEXT:    return $pop[[R]]{{$}}
  %a = fptoui <8 x half> %x to <8 x i16>
  ret <8 x i16> %a
}

define <8 x i16> @trunc_sat_s_v8i16_sat(<8 x half> %x) {
; CHECK-LABEL: trunc_sat_s_v8i16_sat:
; CHECK:         .functype trunc_sat_s_v8i16_sat (v128) -> (v128)
; CHECK-NEXT:    i16x8.trunc_sat_f16x8_s $push0=, $0
; CHECK-NEXT:    return $pop[[R]]{{$}}
  %a = call <8 x i16> @llvm.fptosi.sat.v8i16.v8f16(<8 x half> %x)
  ret <8 x i16> %a
}

define <8 x i16> @trunc_sat_u_v8i16_sat(<8 x half> %x) {
; CHECK-LABEL: trunc_sat_u_v8i16_sat:
; CHECK:         .functype trunc_sat_u_v8i16_sat (v128) -> (v128)
; CHECK-NEXT:    i16x8.trunc_sat_f16x8_u $push0=, $0
; CHECK-NEXT:    return $pop[[R]]{{$}}
  %a = call <8 x i16> @llvm.fptoui.sat.v8i16.v8f16(<8 x half> %x)
  ret <8 x i16> %a
}

; ==============================================================================
; Load and Store
; ==============================================================================
define <8 x half> @load_v8f16(ptr %p) {
; CHECK-LABEL: load_v8f16:
; CHECK:         .functype load_v8f16 (i32) -> (v128)
; CHECK-NEXT:    v128.load $push0=, 0($0)
; CHECK-NEXT:    return $pop0
  %v = load <8 x half>, ptr %p
  ret <8 x half> %v
}

define void @store_v8f16(<8 x half> %v, ptr %p) {
; CHECK-LABEL: store_v8f16:
; CHECK:         .functype store_v8f16 (v128, i32) -> ()
; CHECK-NEXT:    v128.store 0($1), $0
; CHECK-NEXT:    return
  store <8 x half> %v , ptr %p
  ret void
}

; ==============================================================================
; Shuffle
; ==============================================================================
define <8 x half> @shuffle_v8f16(<8 x half> %x, <8 x half> %y) {
; CHECK-LABEL: shuffle_v8f16:
; CHECK:         .functype shuffle_v8f16 (v128, v128) -> (v128)
; CHECK-NEXT:    i8x16.shuffle $push0=, $0, $1, 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31
; CHECK-NEXT:    return $pop0
  %res = shufflevector <8 x half> %x, <8 x half> %y,
    <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
  ret <8 x half> %res
}

define <8 x half> @shuffle_poison_v8f16(<8 x half> %x, <8 x half> %y) {
; CHECK-LABEL: shuffle_poison_v8f16:
; CHECK:         .functype shuffle_poison_v8f16 (v128, v128) -> (v128)
; CHECK-NEXT:    i8x16.shuffle $push0=, $0, $0, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK-NEXT:    return $pop0
  %res = shufflevector <8 x half> %x, <8 x half> %y,
    <8 x i32> <i32 1, i32 poison, i32 poison, i32 poison,
               i32 poison, i32 poison, i32 poison, i32 poison>
  ret <8 x half> %res
}
