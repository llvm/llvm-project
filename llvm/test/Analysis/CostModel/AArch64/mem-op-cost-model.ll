; Check memory cost model action for fixed vector SVE and Neon
; Vector bits size lower than 256 bits end up assuming Neon cost model
; CHECK-NEON has same performance as CHECK-SVE-128

; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+neon  < %s | FileCheck %s --check-prefix=CHECK-NEON
; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=128 < %s | FileCheck %s --check-prefix=CHECK-SVE-128
; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=256 < %s | FileCheck %s --check-prefix=CHECK-SVE-256
; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=512 < %s | FileCheck %s --check-prefix=CHECK-SVE-512

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define <16 x i8> @load16(ptr %ptr) {
; CHECK: function 'load16'
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <16 x i8>, ptr %ptr
  ret <16 x i8> %out
}

define void @store16(ptr %ptr, <16 x i8> %val) {
; CHECK: function 'store16'
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  store <16 x i8> %val, ptr %ptr
  ret void
}

define <8 x i8> @load8(ptr %ptr) {
; CHECK: function 'load8'
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <8 x i8>, ptr %ptr
  ret <8 x i8> %out
}

define void @store8(ptr %ptr, <8 x i8> %val) {
; CHECK: function 'store8'
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  store <8 x i8> %val, ptr %ptr
  ret void
}

define <4 x i8> @load4(ptr %ptr) {
; CHECK: function 'load4'
; CHECK-NEON: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <4 x i8>, ptr %ptr
  ret <4 x i8> %out
}

define void @store4(ptr %ptr, <4 x i8> %val) {
; CHECK: function 'store4'
; CHECK-NEON: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  store <4 x i8> %val, ptr %ptr
  ret void
}

define <16 x i16> @load_256(ptr %ptr) {
; CHECK: function 'load_256'
; CHECK-NEON: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <16 x i16>, ptr %ptr
  ret <16 x i16> %out
}

define <8 x i64> @load_512(ptr %ptr) {
; CHECK: function 'load_512'
; CHECK-NEON: Cost Model: Found an estimated cost of 4 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 4 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <8 x i64>, ptr %ptr
  ret <8 x i64> %out
}

declare <4 x i8> @llvm.masked.gather.v4i8.v4p0(<4 x ptr>, i32 immarg, <4 x i1>, <4 x i8>)
define <4 x i8> @gather_load_4xi8_constant_mask(<4 x ptr> %ptrs) {
; CHECK:         gather_load_4xi8_constant_mask
; CHECK-NEON:    Cost Model: Found an estimated cost of 17 for instruction:  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0
; CHECK-SVE-128: Cost Model: Found an estimated cost of 17 for instruction:  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0
; CHECK-SVE-256: Cost Model: Found an estimated cost of 40 for instruction:  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0
; CHECK-SVE-512: Cost Model: Found an estimated cost of 40 for instruction:  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0
;
  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0(<4 x ptr> %ptrs, i32 1, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i8> undef)
  ret <4 x i8> %lv
}

define <4 x i8> @gather_load_4xi8_variable_mask(<4 x ptr> %ptrs, <4 x i1> %cond) {
; CHECK:         gather_load_4xi8_variable_mask
; CHECK-NEON:    Cost Model: Found an estimated cost of 29 for instruction:  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0
; CHECK-SVE-128: Cost Model: Found an estimated cost of 29 for instruction:  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0
; CHECK-SVE-256: Cost Model: Found an estimated cost of 40 for instruction:  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0
; CHECK-SVE-512: Cost Model: Found an estimated cost of 40 for instruction:  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0
;
  %lv = call <4 x i8> @llvm.masked.gather.v4i8.v4p0(<4 x ptr> %ptrs, i32 1, <4 x i1> %cond, <4 x i8> undef)
  ret <4 x i8> %lv
}

declare void @llvm.masked.scatter.v4i8.v4p0(<4 x i8>, <4 x ptr>, i32 immarg, <4 x i1>)
define void @scatter_store_4xi8_constant_mask(<4 x i8> %val, <4 x ptr> %ptrs) {
; CHECK:         scatter_store_4xi8_constant_mask
; CHECK-NEON:    Cost Model: Found an estimated cost of 17 for instruction:  call void @llvm.masked.scatter.v4i8.v4p0(
; CHECK-SVE-128: Cost Model: Found an estimated cost of 17 for instruction:  call void @llvm.masked.scatter.v4i8.v4p0(
; CHECK-SVE-256: Cost Model: Found an estimated cost of 40 for instruction:  call void @llvm.masked.scatter.v4i8.v4p0(
; CHECK-SVE-512: Cost Model: Found an estimated cost of 40 for instruction:  call void @llvm.masked.scatter.v4i8.v4p0(
;
  call void @llvm.masked.scatter.v4i8.v4p0(<4 x i8> %val, <4 x ptr> %ptrs, i32 1, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
  ret void
}

define void @scatter_store_4xi8_variable_mask(<4 x i8> %val, <4 x ptr> %ptrs, <4 x i1> %cond) {
; CHECK:         scatter_store_4xi8_variable_mask
; CHECK-NEON:    Cost Model: Found an estimated cost of 29 for instruction:  call void @llvm.masked.scatter.v4i8.v4p0(
; CHECK-SVE-128: Cost Model: Found an estimated cost of 29 for instruction:  call void @llvm.masked.scatter.v4i8.v4p0(
; CHECK-SVE-256: Cost Model: Found an estimated cost of 40 for instruction:  call void @llvm.masked.scatter.v4i8.v4p0(
; CHECK-SVE-512: Cost Model: Found an estimated cost of 40 for instruction:  call void @llvm.masked.scatter.v4i8.v4p0(
;
  call void @llvm.masked.scatter.v4i8.v4p0(<4 x i8> %val, <4 x ptr> %ptrs, i32 1, <4 x i1> %cond)
  ret void
}

declare <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr>, i32 immarg, <4 x i1>, <4 x i32>)
define <4 x i32> @gather_load_4xi32_constant_mask(<4 x ptr> %ptrs) {
; CHECK:         gather_load_4xi32_constant_mask
; CHECK-NEON:    Cost Model: Found an estimated cost of 17 for instruction:  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0
; CHECK-SVE-128: Cost Model: Found an estimated cost of 17 for instruction:  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0
; CHECK-SVE-256: Cost Model: Found an estimated cost of 40 for instruction:  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0
; CHECK-SVE-512: Cost Model: Found an estimated cost of 40 for instruction:  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0
;
  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> %ptrs, i32 1, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> undef)
  ret <4 x i32> %lv
}

define <4 x i32> @gather_load_4xi32_variable_mask(<4 x ptr> %ptrs, <4 x i1> %cond) {
; CHECK:         gather_load_4xi32_variable_mask
; CHECK-NEON:    Cost Model: Found an estimated cost of 29 for instruction:  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0
; CHECK-SVE-128: Cost Model: Found an estimated cost of 29 for instruction:  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0
; CHECK-SVE-256: Cost Model: Found an estimated cost of 40 for instruction:  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0
; CHECK-SVE-512: Cost Model: Found an estimated cost of 40 for instruction:  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0
;
  %lv = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> %ptrs, i32 1, <4 x i1> %cond, <4 x i32> undef)
  ret <4 x i32> %lv
}

declare void @llvm.masked.scatter.v4i32.v4p0(<4 x i32>, <4 x ptr>, i32 immarg, <4 x i1>)
define void @scatter_store_4xi32_constant_mask(<4 x i32> %val, <4 x ptr> %ptrs) {
; CHECK:         scatter_store_4xi32_constant_mask
; CHECK-NEON:    Cost Model: Found an estimated cost of 17 for instruction:  call void @llvm.masked.scatter.v4i32.v4p0(
; CHECK-SVE-128: Cost Model: Found an estimated cost of 17 for instruction:  call void @llvm.masked.scatter.v4i32.v4p0(
; CHECK-SVE-256: Cost Model: Found an estimated cost of 40 for instruction:  call void @llvm.masked.scatter.v4i32.v4p0(
; CHECK-SVE-512: Cost Model: Found an estimated cost of 40 for instruction:  call void @llvm.masked.scatter.v4i32.v4p0(
;
  call void @llvm.masked.scatter.v4i32.v4p0(<4 x i32> %val, <4 x ptr> %ptrs, i32 1, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
  ret void
}

define void @scatter_store_4xi32_variable_mask(<4 x i32> %val, <4 x ptr> %ptrs, <4 x i1> %cond) {
; CHECK:         scatter_store_4xi32_variable_mask
; CHECK-NEON:    Cost Model: Found an estimated cost of 29 for instruction:  call void @llvm.masked.scatter.v4i32.v4p0(
; CHECK-SVE-128: Cost Model: Found an estimated cost of 29 for instruction:  call void @llvm.masked.scatter.v4i32.v4p0(
; CHECK-SVE-256: Cost Model: Found an estimated cost of 40 for instruction:  call void @llvm.masked.scatter.v4i32.v4p0(
; CHECK-SVE-512: Cost Model: Found an estimated cost of 40 for instruction:  call void @llvm.masked.scatter.v4i32.v4p0(
;
  call void @llvm.masked.scatter.v4i32.v4p0(<4 x i32> %val, <4 x ptr> %ptrs, i32 1, <4 x i1> %cond)
  ret void
}

declare <256 x i16> @llvm.masked.gather.v256i16.v256p0(<256 x ptr>, i32, <256 x i1>, <256 x i16>)
define void @sve_gather_vls(<256 x i1> %v256i1mask) {
; CHECK-LABEL: 'sve_scatter_vls'
; CHECK-NEON: Cost Model: Found an estimated cost of 1952 for instruction: %res.v256i16 = call <256 x i16> @llvm.masked.gather.v256i16.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x i16> zeroinitializer)
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1952 for instruction: %res.v256i16 = call <256 x i16> @llvm.masked.gather.v256i16.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x i16> zeroinitializer)
; CHECK-SVE-256: Cost Model: Found an estimated cost of 2560 for instruction: %res.v256i16 = call <256 x i16> @llvm.masked.gather.v256i16.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x i16> zeroinitializer)
; CHECK-SVE-512: Cost Model: Found an estimated cost of 2560 for instruction: %res.v256i16 = call <256 x i16> @llvm.masked.gather.v256i16.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x i16> zeroinitializer)
entry:
  %res.v256i16 = call <256 x i16> @llvm.masked.gather.v256i16.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x i16> zeroinitializer)
  ret void
}

declare <256 x float> @llvm.masked.gather.v256f32.v256p0(<256 x ptr>, i32, <256 x i1>, <256 x float>)
define void @sve_gather_vls_float(<256 x i1> %v256i1mask) {
; CHECK-LABEL: 'sve_gather_vls_float'
; CHECK-NEON: Cost Model: Found an estimated cost of 1856 for instruction: %res.v256f32 = call <256 x float> @llvm.masked.gather.v256f32.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x float> zeroinitializer)
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1856 for instruction: %res.v256f32 = call <256 x float> @llvm.masked.gather.v256f32.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x float> zeroinitializer)
; CHECK-SVE-256: Cost Model: Found an estimated cost of 2560 for instruction: %res.v256f32 = call <256 x float> @llvm.masked.gather.v256f32.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x float> zeroinitializer)
; CHECK-SVE-512: Cost Model: Found an estimated cost of 2560 for instruction: %res.v256f32 = call <256 x float> @llvm.masked.gather.v256f32.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x float> zeroinitializer)
entry:
  %res.v256f32 = call <256 x float> @llvm.masked.gather.v256f32.v256p0(<256 x ptr> undef, i32 0, <256 x i1> %v256i1mask, <256 x float> zeroinitializer)
  ret void
}

declare void @llvm.masked.scatter.v256i8.v256p0(<256 x i8>, <256 x ptr>, i32, <256 x i1>)
define void @sve_scatter_vls(<256 x i1> %v256i1mask){
; CHECK-LABEL: 'sve_scatter_vls'
; CHECK-NEON: Cost Model: Found an estimated cost of 2000 for instruction: call void @llvm.masked.scatter.v256i8.v256p0(<256 x i8> undef, <256 x ptr> undef, i32 0, <256 x i1> %v256i1mask)
; CHECK-SVE-128: Cost Model: Found an estimated cost of 2000 for instruction: call void @llvm.masked.scatter.v256i8.v256p0(<256 x i8> undef, <256 x ptr> undef, i32 0, <256 x i1> %v256i1mask)
; CHECK-SVE-256: Cost Model: Found an estimated cost of 2560 for instruction: call void @llvm.masked.scatter.v256i8.v256p0(<256 x i8> undef, <256 x ptr> undef, i32 0, <256 x i1> %v256i1mask)
; CHECK-SVE-512: Cost Model: Found an estimated cost of 2560 for instruction: call void @llvm.masked.scatter.v256i8.v256p0(<256 x i8> undef, <256 x ptr> undef, i32 0, <256 x i1> %v256i1mask)
entry:
  call void @llvm.masked.scatter.v256i8.v256p0(<256 x i8> undef, <256 x ptr> undef, i32 0, <256 x i1> %v256i1mask)
  ret void
}

declare void @llvm.masked.scatter.v512f16.v512p0(<512 x half>, <512 x ptr>, i32, <512 x i1>)
define void @sve_scatter_vls_float(<512 x i1> %v512i1mask){
; CHECK-LABEL: 'sve_scatter_vls_float'
; CHECK-NEON: Cost Model: Found an estimated cost of 3904 for instruction: call void @llvm.masked.scatter.v512f16.v512p0(<512 x half> undef, <512 x ptr> undef, i32 0, <512 x i1> %v512i1mask)
; CHECK-SVE-128: Cost Model: Found an estimated cost of 3904 for instruction: call void @llvm.masked.scatter.v512f16.v512p0(<512 x half> undef, <512 x ptr> undef, i32 0, <512 x i1> %v512i1mask)
; CHECK-SVE-256: Cost Model: Found an estimated cost of 5120 for instruction: call void @llvm.masked.scatter.v512f16.v512p0(<512 x half> undef, <512 x ptr> undef, i32 0, <512 x i1> %v512i1mask)
; CHECK-SVE-512: Cost Model: Found an estimated cost of 5120 for instruction: call void @llvm.masked.scatter.v512f16.v512p0(<512 x half> undef, <512 x ptr> undef, i32 0, <512 x i1> %v512i1mask)
  call void @llvm.masked.scatter.v512f16.v512p0(<512 x half> undef, <512 x ptr> undef, i32 0, <512 x i1> %v512i1mask)
  ret void
}
