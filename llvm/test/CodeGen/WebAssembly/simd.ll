; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128,+sign-ext | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128,+sign-ext | FileCheck %s --check-prefixes CHECK,SIMD128-VM
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test that basic SIMD128 vector manipulation operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: const_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-VM-NOT: v128.const
; SIMD128-NEXT: .functype const_v16i8 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=,
; SIMD128-SAME: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @const_v16i8() {
  ret <16 x i8> <i8 00, i8 01, i8 02, i8 03, i8 04, i8 05, i8 06, i8 07,
                 i8 08, i8 09, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>
}

; CHECK-LABEL: splat_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype splat_v16i8 (i32) -> (v128){{$}}
; SIMD128-NEXT: i8x16.splat $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @splat_v16i8(i8 %x) {
  %v = insertelement <16 x i8> undef, i8 %x, i32 0
  %res = shufflevector <16 x i8> %v, <16 x i8> undef,
    <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0,
                i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i8> %res
}

; CHECK-LABEL: const_splat_v16i8:
; SIMD128: v128.const $push0=, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42{{$}}
define <16 x i8> @const_splat_v16i8() {
  ret <16 x i8> <i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42,
                 i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42>
}

; CHECK-LABEL: extract_v16i8_s:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_v16i8_s (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.extract_lane_s $push[[R:[0-9]+]]=, $0, 13{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_v16i8_s(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 13
  %a = sext i8 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_var_v16i8_s:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_var_v16i8_s (v128, i32) -> (i32){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 15
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]
; SIMD128-NEXT: i32.or $push[[L6:[0-9]+]]=, $2, $pop[[L5]]
; SIMD128-NEXT: i32.load8_s $push[[R:[0-9]+]]=, 0($pop[[L6]])
; SIMD128-NEXT: return $pop[[R]]
define i32 @extract_var_v16i8_s(<16 x i8> %v, i32 %i) {
  %elem = extractelement <16 x i8> %v, i32 %i
  %a = sext i8 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_undef_v16i8_s:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_undef_v16i8_s (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.extract_lane_s $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_undef_v16i8_s(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 undef
  %a = sext i8 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_v16i8_u:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_v16i8_u (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[R:[0-9]+]]=, $0, 13{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_v16i8_u(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 13
  %a = zext i8 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_var_v16i8_u:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_var_v16i8_u (v128, i32) -> (i32){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.or $push[[L6:[0-9]+]]=, $2, $pop[[L5]]{{$}}
; SIMD128-NEXT: i32.load8_u $push[[R:[0-9]+]]=, 0($pop[[L6]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_var_v16i8_u(<16 x i8> %v, i32 %i) {
  %elem = extractelement <16 x i8> %v, i32 %i
  %a = zext i8 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_undef_v16i8_u:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_undef_v16i8_u (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_undef_v16i8_u(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 undef
  %a = zext i8 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_v16i8 (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[R:[0-9]+]]=, $0, 13{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i8 @extract_v16i8(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 13
  ret i8 %elem
}

; CHECK-LABEL: extract_var_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_var_v16i8 (v128, i32) -> (i32){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.or $push[[L6:[0-9]+]]=, $2, $pop[[L5]]{{$}}
; SIMD128-NEXT: i32.load8_u $push[[R:[0-9]+]]=, 0($pop[[L6]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i8 @extract_var_v16i8(<16 x i8> %v, i32 %i) {
  %elem = extractelement <16 x i8> %v, i32 %i
  ret i8 %elem
}

; CHECK-LABEL: extract_undef_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype extract_undef_v16i8 (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i8 @extract_undef_v16i8(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 undef
  ret i8 %elem
}

; CHECK-LABEL: replace_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype replace_v16i8 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[R:[0-9]+]]=, $0, 11, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @replace_v16i8(<16 x i8> %v, i8 %x) {
  %res = insertelement <16 x i8> %v, i8 %x, i32 11
  ret <16 x i8> %res
}

; CHECK-LABEL: replace_var_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype replace_var_v16i8 (v128, i32, i32) -> (v128){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $3=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.or $push[[L6:[0-9]+]]=, $3, $pop[[L5]]{{$}}
; SIMD128-NEXT: i32.store8 0($pop[[L6]]), $2{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($3){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @replace_var_v16i8(<16 x i8> %v, i32 %i, i8 %x) {
  %res = insertelement <16 x i8> %v, i8 %x, i32 %i
  ret <16 x i8> %res
}

; CHECK-LABEL: replace_zero_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype replace_zero_v16i8 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[R:[0-9]+]]=, $0, 0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @replace_zero_v16i8(<16 x i8> %v, i8 %x) {
  %res = insertelement <16 x i8> %v, i8 %x, i32 0
  ret <16 x i8> %res
}

; CHECK-LABEL: shuffle_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $1,
; SIMD128-SAME: 0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shuffle_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %res = shufflevector <16 x i8> %x, <16 x i8> %y,
    <16 x i32> <i32 0, i32 17, i32 2, i32 19, i32 4, i32 21, i32 6, i32 23,
                i32 8, i32 25, i32 10, i32 27, i32 12, i32 29, i32 14, i32 31>
  ret <16 x i8> %res
}

; CHECK-LABEL: shuffle_undef_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_undef_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $0,
; SIMD128-SAME: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shuffle_undef_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %res = shufflevector <16 x i8> %x, <16 x i8> %y,
    <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %res
}

; CHECK-LABEL: build_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype build_v16i8 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (v128){{$}}
; SIMD128-NEXT: i8x16.splat $push[[L0:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L1:[0-9]+]]=, $pop[[L0]], 1, $1{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L2:[0-9]+]]=, $pop[[L1]], 2, $2{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L3:[0-9]+]]=, $pop[[L2]], 3, $3{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L4:[0-9]+]]=, $pop[[L3]], 4, $4{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L5:[0-9]+]]=, $pop[[L4]], 5, $5{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L6:[0-9]+]]=, $pop[[L5]], 6, $6{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L7:[0-9]+]]=, $pop[[L6]], 7, $7{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L8:[0-9]+]]=, $pop[[L7]], 8, $8{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L9:[0-9]+]]=, $pop[[L8]], 9, $9{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L10:[0-9]+]]=, $pop[[L9]], 10, $10{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L11:[0-9]+]]=, $pop[[L10]], 11, $11{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L12:[0-9]+]]=, $pop[[L11]], 12, $12{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L13:[0-9]+]]=, $pop[[L12]], 13, $13{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[L14:[0-9]+]]=, $pop[[L13]], 14, $14{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[R:[0-9]+]]=, $pop[[L14]], 15, $15{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @build_v16i8(i8 %x0, i8 %x1, i8 %x2, i8 %x3,
                              i8 %x4, i8 %x5, i8 %x6, i8 %x7,
                              i8 %x8, i8 %x9, i8 %x10, i8 %x11,
                              i8 %x12, i8 %x13, i8 %x14, i8 %x15) {
  %t0 = insertelement <16 x i8> undef, i8 %x0, i32 0
  %t1 = insertelement <16 x i8> %t0, i8 %x1, i32 1
  %t2 = insertelement <16 x i8> %t1, i8 %x2, i32 2
  %t3 = insertelement <16 x i8> %t2, i8 %x3, i32 3
  %t4 = insertelement <16 x i8> %t3, i8 %x4, i32 4
  %t5 = insertelement <16 x i8> %t4, i8 %x5, i32 5
  %t6 = insertelement <16 x i8> %t5, i8 %x6, i32 6
  %t7 = insertelement <16 x i8> %t6, i8 %x7, i32 7
  %t8 = insertelement <16 x i8> %t7, i8 %x8, i32 8
  %t9 = insertelement <16 x i8> %t8, i8 %x9, i32 9
  %t10 = insertelement <16 x i8> %t9, i8 %x10, i32 10
  %t11 = insertelement <16 x i8> %t10, i8 %x11, i32 11
  %t12 = insertelement <16 x i8> %t11, i8 %x12, i32 12
  %t13 = insertelement <16 x i8> %t12, i8 %x13, i32 13
  %t14 = insertelement <16 x i8> %t13, i8 %x14, i32 14
  %res = insertelement <16 x i8> %t14, i8 %x15, i32 15
  ret <16 x i8> %res
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: const_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-VM-NOT: v128.const
; SIMD128-NEXT: .functype const_v8i16 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=, 256, 770, 1284, 1798, 2312, 2826, 3340, 3854{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @const_v8i16() {
  ret <8 x i16> <i16 256, i16 770, i16 1284, i16 1798,
                 i16 2312, i16 2826, i16 3340, i16 3854>
}

; CHECK-LABEL: splat_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype splat_v8i16 (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.splat $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @splat_v8i16(i16 %x) {
  %v = insertelement <8 x i16> undef, i16 %x, i32 0
  %res = shufflevector <8 x i16> %v, <8 x i16> undef,
    <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %res
}

; CHECK-LABEL: const_splat_v8i16:
; SIMD128: v128.const $push0=, 42, 42, 42, 42, 42, 42, 42, 42{{$}}
define <8 x i16> @const_splat_v8i16() {
  ret <8 x i16> <i16 42, i16 42, i16 42, i16 42, i16 42, i16 42, i16 42, i16 42>
}

; CHECK-LABEL: extract_v8i16_s:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_v8i16_s (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.extract_lane_s $push[[R:[0-9]+]]=, $0, 5{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_v8i16_s(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 5
  %a = sext i16 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_var_v8i16_s:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_var_v8i16_s (v128, i32) -> (i32){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L8:[0-9]+]]=, $2, $pop[[L7]]{{$}}
; SIMD128-NEXT: i32.load16_s $push[[R:[0-9]+]]=, 0($pop[[L8]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_var_v8i16_s(<8 x i16> %v, i32 %i) {
  %elem = extractelement <8 x i16> %v, i32 %i
  %a = sext i16 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_undef_v8i16_s:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_undef_v8i16_s (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.extract_lane_s $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_undef_v8i16_s(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 undef
  %a = sext i16 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_v8i16_u:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_v8i16_u (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[R:[0-9]+]]=, $0, 5{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_v8i16_u(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 5
  %a = zext i16 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_var_v8i16_u:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_var_v8i16_u (v128, i32) -> (i32){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L8:[0-9]+]]=, $2, $pop[[L7]]{{$}}
; SIMD128-NEXT: i32.load16_u $push[[R:[0-9]+]]=, 0($pop[[L8]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_var_v8i16_u(<8 x i16> %v, i32 %i) {
  %elem = extractelement <8 x i16> %v, i32 %i
  %a = zext i16 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_undef_v8i16_u:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_undef_v8i16_u (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_undef_v8i16_u(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 undef
  %a = zext i16 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_v8i16 (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[R:[0-9]+]]=, $0, 5{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i16 @extract_v8i16(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 5
  ret i16 %elem
}

; CHECK-LABEL: extract_var_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_var_v8i16 (v128, i32) -> (i32){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L8:[0-9]+]]=, $2, $pop[[L7]]{{$}}
; SIMD128-NEXT: i32.load16_u $push[[R:[0-9]+]]=, 0($pop[[L8]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i16 @extract_var_v8i16(<8 x i16> %v, i32 %i) {
  %elem = extractelement <8 x i16> %v, i32 %i
  ret i16 %elem
}

; CHECK-LABEL: extract_undef_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extract_undef_v8i16 (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i16 @extract_undef_v8i16(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 undef
  ret i16 %elem
}

; CHECK-LABEL: replace_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype replace_v8i16 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[R:[0-9]+]]=, $0, 7, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @replace_v8i16(<8 x i16> %v, i16 %x) {
  %res = insertelement <8 x i16> %v, i16 %x, i32 7
  ret <8 x i16> %res
}

; CHECK-LABEL: replace_var_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype replace_var_v8i16 (v128, i32, i32) -> (v128){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $3=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L8:[0-9]+]]=, $3, $pop[[L7]]{{$}}
; SIMD128-NEXT: i32.store16 0($pop[[L8]]), $2{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($3){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @replace_var_v8i16(<8 x i16> %v, i32 %i, i16 %x) {
  %res = insertelement <8 x i16> %v, i16 %x, i32 %i
  ret <8 x i16> %res
}

; CHECK-LABEL: replace_zero_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype replace_zero_v8i16 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[R:[0-9]+]]=, $0, 0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @replace_zero_v8i16(<8 x i16> %v, i16 %x) {
  %res = insertelement <8 x i16> %v, i16 %x, i32 0
  ret <8 x i16> %res
}

; CHECK-LABEL: shuffle_v8i16:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $1,
; SIMD128-SAME: 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shuffle_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %res = shufflevector <8 x i16> %x, <8 x i16> %y,
    <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
  ret <8 x i16> %res
}

; CHECK-LABEL: shuffle_undef_v8i16:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_undef_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $0,
; SIMD128-SAME: 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shuffle_undef_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %res = shufflevector <8 x i16> %x, <8 x i16> %y,
    <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef,
               i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %res
}

; CHECK-LABEL: build_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype build_v8i16 (i32, i32, i32, i32, i32, i32, i32, i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.splat $push[[L0:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[L1:[0-9]+]]=, $pop[[L0]], 1, $1{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[L2:[0-9]+]]=, $pop[[L1]], 2, $2{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[L3:[0-9]+]]=, $pop[[L2]], 3, $3{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[L4:[0-9]+]]=, $pop[[L3]], 4, $4{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[L5:[0-9]+]]=, $pop[[L4]], 5, $5{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[L6:[0-9]+]]=, $pop[[L5]], 6, $6{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[R:[0-9]+]]=, $pop[[L6]], 7, $7{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @build_v8i16(i16 %x0, i16 %x1, i16 %x2, i16 %x3,
                              i16 %x4, i16 %x5, i16 %x6, i16 %x7) {
  %t0 = insertelement <8 x i16> undef, i16 %x0, i32 0
  %t1 = insertelement <8 x i16> %t0, i16 %x1, i32 1
  %t2 = insertelement <8 x i16> %t1, i16 %x2, i32 2
  %t3 = insertelement <8 x i16> %t2, i16 %x3, i32 3
  %t4 = insertelement <8 x i16> %t3, i16 %x4, i32 4
  %t5 = insertelement <8 x i16> %t4, i16 %x5, i32 5
  %t6 = insertelement <8 x i16> %t5, i16 %x6, i32 6
  %res = insertelement <8 x i16> %t6, i16 %x7, i32 7
  ret <8 x i16> %res
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: const_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-VM-NOT: v128.const
; SIMD128-NEXT: .functype const_v4i32 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=, 50462976, 117835012, 185207048, 252579084{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @const_v4i32() {
  ret <4 x i32> <i32 50462976, i32 117835012, i32 185207048, i32 252579084>
}

; CHECK-LABEL: splat_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype splat_v4i32 (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.splat $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @splat_v4i32(i32 %x) {
  %v = insertelement <4 x i32> undef, i32 %x, i32 0
  %res = shufflevector <4 x i32> %v, <4 x i32> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  ret <4 x i32> %res
}

; CHECK-LABEL: const_splat_v4i32:
; SIMD128: v128.const $push0=, 42, 42, 42, 42{{$}}
define <4 x i32> @const_splat_v4i32() {
  ret <4 x i32> <i32 42, i32 42, i32 42, i32 42>
}

; CHECK-LABEL: extract_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype extract_v4i32 (v128) -> (i32){{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[R:[0-9]+]]=, $0, 3{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_v4i32(<4 x i32> %v) {
  %elem = extractelement <4 x i32> %v, i32 3
  ret i32 %elem
}

; CHECK-LABEL: extract_var_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype extract_var_v4i32 (v128, i32) -> (i32){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 3{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 2{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L4:[0-9]+]]=, $2, $pop[[L7]]{{$}}
; SIMD128-NEXT: i32.load $push[[R:[0-9]+]]=, 0($pop[[L4]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_var_v4i32(<4 x i32> %v, i32 %i) {
  %elem = extractelement <4 x i32> %v, i32 %i
  ret i32 %elem
}

; CHECK-LABEL: extract_zero_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype extract_zero_v4i32 (v128) -> (i32){{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @extract_zero_v4i32(<4 x i32> %v) {
  %elem = extractelement <4 x i32> %v, i32 0
  ret i32 %elem
}

; CHECK-LABEL: replace_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype replace_v4i32 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.replace_lane $push[[R:[0-9]+]]=, $0, 2, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @replace_v4i32(<4 x i32> %v, i32 %x) {
  %res = insertelement <4 x i32> %v, i32 %x, i32 2
  ret <4 x i32> %res
}

; CHECK-LABEL: replace_var_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype replace_var_v4i32 (v128, i32, i32) -> (v128){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $3=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L4:[0-9]+]]=, 3{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L4]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 2{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L4:[0-9]+]]=, $3, $pop[[L7]]{{$}}
; SIMD128-NEXT: i32.store 0($pop[[L4]]), $2{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($3){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @replace_var_v4i32(<4 x i32> %v, i32 %i, i32 %x) {
  %res = insertelement <4 x i32> %v, i32 %x, i32 %i
  ret <4 x i32> %res
}

; CHECK-LABEL: replace_zero_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype replace_zero_v4i32 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.replace_lane $push[[R:[0-9]+]]=, $0, 0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @replace_zero_v4i32(<4 x i32> %v, i32 %x) {
  %res = insertelement <4 x i32> %v, i32 %x, i32 0
  ret <4 x i32> %res
}

; CHECK-LABEL: shuffle_v4i32:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $1,
; SIMD128-SAME: 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shuffle_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %res = shufflevector <4 x i32> %x, <4 x i32> %y,
    <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i32> %res
}

; CHECK-LABEL: shuffle_undef_v4i32:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_undef_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $0,
; SIMD128-SAME: 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shuffle_undef_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %res = shufflevector <4 x i32> %x, <4 x i32> %y,
    <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  ret <4 x i32> %res
}

; CHECK-LABEL: build_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype build_v4i32 (i32, i32, i32, i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.splat $push[[L0:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: i32x4.replace_lane $push[[L1:[0-9]+]]=, $pop[[L0]], 1, $1{{$}}
; SIMD128-NEXT: i32x4.replace_lane $push[[L2:[0-9]+]]=, $pop[[L1]], 2, $2{{$}}
; SIMD128-NEXT: i32x4.replace_lane $push[[R:[0-9]+]]=, $pop[[L2]], 3, $3{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @build_v4i32(i32 %x0, i32 %x1, i32 %x2, i32 %x3) {
  %t0 = insertelement <4 x i32> undef, i32 %x0, i32 0
  %t1 = insertelement <4 x i32> %t0, i32 %x1, i32 1
  %t2 = insertelement <4 x i32> %t1, i32 %x2, i32 2
  %res = insertelement <4 x i32> %t2, i32 %x3, i32 3
  ret <4 x i32> %res
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: const_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-VM-NOT: v128.const
; SIMD128-NEXT: .functype const_v2i64 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=, 506097522914230528, 1084818905618843912{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @const_v2i64() {
  ret <2 x i64> <i64 506097522914230528, i64 1084818905618843912>
}

; CHECK-LABEL: splat_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype splat_v2i64 (i64) -> (v128){{$}}
; SIMD128-NEXT: i64x2.splat $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @splat_v2i64(i64 %x) {
  %t1 = insertelement <2 x i64> zeroinitializer, i64 %x, i32 0
  %res = insertelement <2 x i64> %t1, i64 %x, i32 1
  ret <2 x i64> %res
}

; CHECK-LABEL: const_splat_v2i64:
; SIMD128: v128.const $push0=, 42, 42{{$}}
define <2 x i64> @const_splat_v2i64() {
  ret <2 x i64> <i64 42, i64 42>
}

; CHECK-LABEL: extract_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype extract_v2i64 (v128) -> (i64){{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[R:[0-9]+]]=, $0, 1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i64 @extract_v2i64(<2 x i64> %v) {
  %elem = extractelement <2 x i64> %v, i64 1
  ret i64 %elem
}

; CHECK-LABEL: extract_var_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype extract_var_v2i64 (v128, i32) -> (i64){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L2:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L2]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 3{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L2:[0-9]+]]=, $2, $pop[[L7]]{{$}}
; SIMD128-NEXT: i64.load $push[[R:[0-9]+]]=, 0($pop[[L2]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i64 @extract_var_v2i64(<2 x i64> %v, i32 %i) {
  %elem = extractelement <2 x i64> %v, i32 %i
  ret i64 %elem
}

; CHECK-LABEL: extract_zero_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype extract_zero_v2i64 (v128) -> (i64){{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i64 @extract_zero_v2i64(<2 x i64> %v) {
  %elem = extractelement <2 x i64> %v, i64 0
  ret i64 %elem
}

; CHECK-LABEL: replace_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype replace_v2i64 (v128, i64) -> (v128){{$}}
; SIMD128-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $0, 0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @replace_v2i64(<2 x i64> %v, i64 %x) {
  %res = insertelement <2 x i64> %v, i64 %x, i32 0
  ret <2 x i64> %res
}

; CHECK-LABEL: replace_var_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype replace_var_v2i64 (v128, i32, i64) -> (v128){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $3=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L2:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L2]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 3{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L2:[0-9]+]]=, $3, $pop[[L7]]{{$}}
; SIMD128-NEXT: i64.store 0($pop[[L2]]), $2{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($3){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @replace_var_v2i64(<2 x i64> %v, i32 %i, i64 %x) {
  %res = insertelement <2 x i64> %v, i64 %x, i32 %i
  ret <2 x i64> %res
}

; CHECK-LABEL: replace_zero_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype replace_zero_v2i64 (v128, i64) -> (v128){{$}}
; SIMD128-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $0, 0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @replace_zero_v2i64(<2 x i64> %v, i64 %x) {
  %res = insertelement <2 x i64> %v, i64 %x, i32 0
  ret <2 x i64> %res
}

; CHECK-LABEL: shuffle_v2i64:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $1,
; SIMD128-SAME: 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shuffle_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %res = shufflevector <2 x i64> %x, <2 x i64> %y, <2 x i32> <i32 0, i32 3>
  ret <2 x i64> %res
}

; CHECK-LABEL: shuffle_undef_v2i64:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_undef_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $0,
; SIMD128-SAME: 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shuffle_undef_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %res = shufflevector <2 x i64> %x, <2 x i64> %y,
    <2 x i32> <i32 1, i32 undef>
  ret <2 x i64> %res
}

; CHECK-LABEL: build_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype build_v2i64 (i64, i64) -> (v128){{$}}
; SIMD128-NEXT: i64x2.splat $push[[L0:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $pop[[L0]], 1, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @build_v2i64(i64 %x0, i64 %x1) {
  %t0 = insertelement <2 x i64> undef, i64 %x0, i32 0
  %res = insertelement <2 x i64> %t0, i64 %x1, i32 1
  ret <2 x i64> %res
}

; ==============================================================================
; 4 x f32
; ==============================================================================
; CHECK-LABEL: const_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-VM-NOT: v128.const
; SIMD128-NEXT: .functype const_v4f32 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=,
; SIMD128-SAME: 0x1.0402p-121, 0x1.0c0a08p-113, 0x1.14121p-105, 0x1.1c1a18p-97{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @const_v4f32() {
  ret <4 x float> <float 0x3860402000000000, float 0x38e0c0a080000000,
                   float 0x3961412100000000, float 0x39e1c1a180000000>
}

; CHECK-LABEL: splat_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype splat_v4f32 (f32) -> (v128){{$}}
; SIMD128-NEXT: f32x4.splat $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @splat_v4f32(float %x) {
  %v = insertelement <4 x float> undef, float %x, i32 0
  %res = shufflevector <4 x float> %v, <4 x float> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  ret <4 x float> %res
}

; CHECK-LABEL: const_splat_v4f32
; SIMD128: v128.const $push0=, 0x1.5p5, 0x1.5p5, 0x1.5p5, 0x1.5p5{{$}}
define <4 x float> @const_splat_v4f32() {
  ret <4 x float> <float 42., float 42., float 42., float 42.>
}

; CHECK-LABEL: extract_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype extract_v4f32 (v128) -> (f32){{$}}
; SIMD128-NEXT: f32x4.extract_lane $push[[R:[0-9]+]]=, $0, 3{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define float @extract_v4f32(<4 x float> %v) {
  %elem = extractelement <4 x float> %v, i32 3
  ret float %elem
}

; CHECK-LABEL: extract_var_v4f32:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype extract_var_v4f32 (v128, i32) -> (f32){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L2:[0-9]+]]=, 3{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L2]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 2{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L2:[0-9]+]]=, $2, $pop[[L7]]{{$}}
; SIMD128-NEXT: f32.load $push[[R:[0-9]+]]=, 0($pop[[L2]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define float @extract_var_v4f32(<4 x float> %v, i32 %i) {
  %elem = extractelement <4 x float> %v, i32 %i
  ret float %elem
}

; CHECK-LABEL: extract_zero_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype extract_zero_v4f32 (v128) -> (f32){{$}}
; SIMD128-NEXT: f32x4.extract_lane $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define float @extract_zero_v4f32(<4 x float> %v) {
  %elem = extractelement <4 x float> %v, i32 0
  ret float %elem
}

; CHECK-LABEL: replace_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype replace_v4f32 (v128, f32) -> (v128){{$}}
; SIMD128-NEXT: f32x4.replace_lane $push[[R:[0-9]+]]=, $0, 2, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @replace_v4f32(<4 x float> %v, float %x) {
  %res = insertelement <4 x float> %v, float %x, i32 2
  ret <4 x float> %res
}

; CHECK-LABEL: replace_var_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype replace_var_v4f32 (v128, i32, f32) -> (v128){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $3=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L2:[0-9]+]]=, 3{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L2]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 2{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L2:[0-9]+]]=, $3, $pop[[L7]]{{$}}
; SIMD128-NEXT: f32.store 0($pop[[L2]]), $2{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($3){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @replace_var_v4f32(<4 x float> %v, i32 %i, float %x) {
  %res = insertelement <4 x float> %v, float %x, i32 %i
  ret <4 x float> %res
}

; CHECK-LABEL: replace_zero_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype replace_zero_v4f32 (v128, f32) -> (v128){{$}}
; SIMD128-NEXT: f32x4.replace_lane $push[[R:[0-9]+]]=, $0, 0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @replace_zero_v4f32(<4 x float> %v, float %x) {
  %res = insertelement <4 x float> %v, float %x, i32 0
  ret <4 x float> %res
}

; CHECK-LABEL: shuffle_v4f32:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $1,
; SIMD128-SAME: 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @shuffle_v4f32(<4 x float> %x, <4 x float> %y) {
  %res = shufflevector <4 x float> %x, <4 x float> %y,
    <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %res
}

; CHECK-LABEL: shuffle_undef_v4f32:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_undef_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $0,
; SIMD128-SAME: 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @shuffle_undef_v4f32(<4 x float> %x, <4 x float> %y) {
  %res = shufflevector <4 x float> %x, <4 x float> %y,
    <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  ret <4 x float> %res
}

; CHECK-LABEL: build_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype build_v4f32 (f32, f32, f32, f32) -> (v128){{$}}
; SIMD128-NEXT: f32x4.splat $push[[L0:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: f32x4.replace_lane $push[[L1:[0-9]+]]=, $pop[[L0]], 1, $1{{$}}
; SIMD128-NEXT: f32x4.replace_lane $push[[L2:[0-9]+]]=, $pop[[L1]], 2, $2{{$}}
; SIMD128-NEXT: f32x4.replace_lane $push[[R:[0-9]+]]=, $pop[[L2]], 3, $3{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @build_v4f32(float %x0, float %x1, float %x2, float %x3) {
  %t0 = insertelement <4 x float> undef, float %x0, i32 0
  %t1 = insertelement <4 x float> %t0, float %x1, i32 1
  %t2 = insertelement <4 x float> %t1, float %x2, i32 2
  %res = insertelement <4 x float> %t2, float %x3, i32 3
  ret <4 x float> %res
}

; ==============================================================================
; 2 x f64
; ==============================================================================
; CHECK-LABEL: const_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: v128.const
; SIMD128-NEXT: .functype const_v2f64 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=, 0x1.60504030201p-911, 0x1.e0d0c0b0a0908p-783{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @const_v2f64() {
  ret <2 x double> <double 0x0706050403020100, double 0x0F0E0D0C0B0A0908>
}

; CHECK-LABEL: splat_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype splat_v2f64 (f64) -> (v128){{$}}
; SIMD128-NEXT: f64x2.splat $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @splat_v2f64(double %x) {
  %t1 = insertelement <2 x double> zeroinitializer, double %x, i3 0
  %res = insertelement <2 x double> %t1, double %x, i32 1
  ret <2 x double> %res
}

; CHECK-LABEL: const_splat_v2f64:
; SIMD128: v128.const $push0=, 0x1.5p5, 0x1.5p5{{$}}
define <2 x double> @const_splat_v2f64() {
  ret <2 x double> <double 42., double 42.>
}

; CHECK-LABEL: extract_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype extract_v2f64 (v128) -> (f64){{$}}
; SIMD128-NEXT: f64x2.extract_lane $push[[R:[0-9]+]]=, $0, 1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define double @extract_v2f64(<2 x double> %v) {
  %elem = extractelement <2 x double> %v, i32 1
  ret double %elem
}

; CHECK-LABEL: extract_var_v2f64:
; NO-SIMD128-NOT: i62x2
; SIMD128-NEXT: .functype extract_var_v2f64 (v128, i32) -> (f64){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $2=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L2:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L2]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 3{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L2:[0-9]+]]=, $2, $pop[[L7]]{{$}}
; SIMD128-NEXT: f64.load $push[[R:[0-9]+]]=, 0($pop[[L2]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define double @extract_var_v2f64(<2 x double> %v, i32 %i) {
  %elem = extractelement <2 x double> %v, i32 %i
  ret double %elem
}

; CHECK-LABEL: extract_zero_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype extract_zero_v2f64 (v128) -> (f64){{$}}
; SIMD128-NEXT: f64x2.extract_lane $push[[R:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define double @extract_zero_v2f64(<2 x double> %v) {
  %elem = extractelement <2 x double> %v, i32 0
  ret double %elem
}

; CHECK-LABEL: replace_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype replace_v2f64 (v128, f64) -> (v128){{$}}
; SIMD128-NEXT: f64x2.replace_lane $push[[R:[0-9]+]]=, $0, 0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @replace_v2f64(<2 x double> %v, double %x) {
  %res = insertelement <2 x double> %v, double %x, i32 0
  ret <2 x double> %res
}

; CHECK-LABEL: replace_var_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype replace_var_v2f64 (v128, i32, f64) -> (v128){{$}}
; SIMD128-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer{{$}}
; SIMD128-NEXT: i32.const $push[[L1:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: local.tee $push[[L3:[0-9]+]]=, $3=, $pop[[L2]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[L3]]), $0{{$}}
; SIMD128-NEXT: i32.const $push[[L2:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.and $push[[L5:[0-9]+]]=, $1, $pop[[L2]]{{$}}
; SIMD128-NEXT: i32.const $push[[L6:[0-9]+]]=, 3{{$}}
; SIMD128-NEXT: i32.shl $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; SIMD128-NEXT: i32.or $push[[L2:[0-9]+]]=, $3, $pop[[L7]]{{$}}
; SIMD128-NEXT: f64.store 0($pop[[L2]]), $2{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($3){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @replace_var_v2f64(<2 x double> %v, i32 %i, double %x) {
  %res = insertelement <2 x double> %v, double %x, i32 %i
  ret <2 x double> %res
}

; CHECK-LABEL: replace_zero_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype replace_zero_v2f64 (v128, f64) -> (v128){{$}}
; SIMD128-NEXT: f64x2.replace_lane $push[[R:[0-9]+]]=, $0, 0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @replace_zero_v2f64(<2 x double> %v, double %x) {
  %res = insertelement <2 x double> %v, double %x, i32 0
  ret <2 x double> %res
}

; CHECK-LABEL: shuffle_v2f64:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $1,
; SIMD128-SAME: 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @shuffle_v2f64(<2 x double> %x, <2 x double> %y) {
  %res = shufflevector <2 x double> %x, <2 x double> %y,
    <2 x i32> <i32 0, i32 3>
  ret <2 x double> %res
}

; CHECK-LABEL: shuffle_undef_v2f64:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shuffle_undef_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $0,
; SIMD128-SAME: 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @shuffle_undef_v2f64(<2 x double> %x, <2 x double> %y) {
  %res = shufflevector <2 x double> %x, <2 x double> %y,
    <2 x i32> <i32 1, i32 undef>
  ret <2 x double> %res
}

; CHECK-LABEL: build_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype build_v2f64 (f64, f64) -> (v128){{$}}
; SIMD128-NEXT: f64x2.splat $push[[L0:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: f64x2.replace_lane $push[[R:[0-9]+]]=, $pop[[L0]], 1, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @build_v2f64(double %x0, double %x1) {
  %t0 = insertelement <2 x double> undef, double %x0, i32 0
  %res = insertelement <2 x double> %t0, double %x1, i32 1
  ret <2 x double> %res
}
