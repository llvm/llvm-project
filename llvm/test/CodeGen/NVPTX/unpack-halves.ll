; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx60 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,OLD
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,NEW
; RUN: llc < %s -enable-new-pm -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,NEW
; RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | %ptxas-verify -arch=sm_90 %}

; Keep both halves in 16-bit registers, without folding the truncates into
; stores or extending them for the function return ABI.
define void @unpack_lshr(i32 %word) {
; CHECK-LABEL: unpack_lshr(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK-NOT: cvt.
; CHECK: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK-NOT: cvt.
; CHECK: // halves [[LO]], [[HI]]
  %lo = trunc i32 %word to i16
  %shift = lshr i32 %word, 16
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// halves $0, $1", "h,h"(i16 %lo, i16 %hi)
  ret void
}

define void @unpack_ashr(i32 %word) {
; CHECK-LABEL: unpack_ashr(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK-NOT: cvt.
; CHECK: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK: // halves [[LO]], [[HI]]
  %lo = trunc i32 %word to i16
  %shift = ashr i32 %word, 16
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// halves $0, $1", "h,h"(i16 %lo, i16 %hi)
  ret void
}

; Reverse the use order to exercise selection starting from either half.
define void @unpack_reverse(i32 %word) {
; CHECK-LABEL: unpack_reverse(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK-NOT: cvt.
; CHECK: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK: // halves [[HI]], [[LO]]
  %shift = lshr i32 %word, 16
  %hi = trunc i32 %shift to i16
  %lo = trunc i32 %word to i16
  call void asm sideeffect "// halves $0, $1", "h,h"(i16 %hi, i16 %lo)
  ret void
}

; Other uses of the source, shift, and extracted halves must remain valid.
define void @unpack_multiuse(i32 %word) {
; CHECK-LABEL: unpack_multiuse(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK-DAG: shr.u32 [[SHIFT:%r[0-9]+]], [[WORD]], 16;
; CHECK-DAG: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK: // uses [[LO]], [[HI]], [[WORD]], [[SHIFT]], [[LO]], [[HI]]
  %lo = trunc i32 %word to i16
  %shift = lshr i32 %word, 16
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// uses $0, $1, $2, $3, $4, $5", "h,h,r,r,h,h"(i16 %lo, i16 %hi, i32 %word, i32 %shift, i16 %lo, i16 %hi)
  ret void
}

define void @unpack_ashr_multiuse(i32 %word) {
; CHECK-LABEL: unpack_ashr_multiuse(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK-DAG: shr.s32 [[SHIFT:%r[0-9]+]], [[WORD]], 16;
; CHECK-DAG: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK: // uses [[LO]], [[HI]], [[SHIFT]]
  %lo = trunc i32 %word to i16
  %shift = ashr i32 %word, 16
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// uses $0, $1, $2", "h,h,r"(i16 %lo, i16 %hi, i32 %shift)
  ret void
}

define void @low_only(i32 %word) {
; CHECK-LABEL: low_only(
; CHECK: cvt.u16.u32 [[LO:%rs[0-9]+]], %r{{[0-9]+}};
; CHECK: // half [[LO]]
  %lo = trunc i32 %word to i16
  call void asm sideeffect "// half $0, word $1", "h,r"(i16 %lo, i32 %word)
  ret void
}

define void @high_only(i32 %word) {
; CHECK-LABEL: high_only(
; OLD: { .reg .b16 tmp; mov.b32 {tmp, [[HI:%rs[0-9]+]]}, %r{{[0-9]+}}; }
; NEW: mov.b32 {_, [[HI:%rs[0-9]+]]}, %r{{[0-9]+}};
; CHECK: // half [[HI]]
  %shift = lshr i32 %word, 16
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// half $0, word $1", "h,r"(i16 %hi, i32 %word)
  ret void
}

; These fields do not partition a word into two halves.
define void @different_shift(i32 %word) {
; CHECK-LABEL: different_shift(
; CHECK-NOT: mov.b32 {
; CHECK: // halves
  %lo = trunc i32 %word to i16
  %shift = lshr i32 %word, 15
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// halves $0, $1", "h,h"(i16 %lo, i16 %hi)
  ret void
}

define void @variable_shift(i32 %word, i32 %shift) {
; CHECK-LABEL: variable_shift(
; CHECK-NOT: mov.b32 {
; CHECK: shr.u32
; CHECK: // halves
  %lo = trunc i32 %word to i16
  %shifted = lshr i32 %word, %shift
  %hi = trunc i32 %shifted to i16
  call void asm sideeffect "// halves $0, $1", "h,h"(i16 %lo, i16 %hi)
  ret void
}

define void @different_words(i32 %a, i32 %b) {
; CHECK-LABEL: different_words(
; CHECK: cvt.u16.u32
; OLD: { .reg .b16 tmp; mov.b32 {tmp,
; NEW: mov.b32 {_,
; CHECK: // halves
  %lo = trunc i32 %a to i16
  %shift = lshr i32 %b, 16
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// halves $0, $1, words $2, $3", "h,h,r,r"(i16 %lo, i16 %hi, i32 %a, i32 %b)
  ret void
}

; Do not change the selection of 64-bit extractions.
define void @unpack_i64(i64 %word) {
; CHECK-LABEL: unpack_i64(
; CHECK: cvt.u32.u64
; OLD: { .reg .b32 tmp; mov.b64 {tmp,
; NEW: mov.b64 {_,
; CHECK: // halves
  %lo = trunc i64 %word to i32
  %shift = lshr i64 %word, 32
  %hi = trunc i64 %shift to i32
  call void asm sideeffect "// halves $0, $1", "r,r"(i32 %lo, i32 %hi)
  ret void
}

; Scalarized BF16 arithmetic following a bitcast of each half, as in Triton.
define void @unpack_scale_bf16(i32 %word, ptr addrspace(1) %out_lo, ptr addrspace(1) %out_hi) {
; CHECK-LABEL: unpack_scale_bf16(
; NEW: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, %r{{[0-9]+}};
; NEW-DAG: {{(mul|fma)}}.{{.*}}bf16 {{.*}}[[LO]]
; NEW-DAG: {{(mul|fma)}}.{{.*}}bf16 {{.*}}[[HI]]
  %lo = trunc i32 %word to i16
  %shift = lshr i32 %word, 16
  %hi = trunc i32 %shift to i16
  %lo_bf = bitcast i16 %lo to bfloat
  %hi_bf = bitcast i16 %hi to bfloat
  %scaled_lo = fmul bfloat %lo_bf, 0xR7E80
  %scaled_hi = fmul bfloat %hi_bf, 0xR7E80
  store bfloat %scaled_lo, ptr addrspace(1) %out_lo, align 2
  store bfloat %scaled_hi, ptr addrspace(1) %out_hi, align 2
  ret void
}

; These extracts have different vector types, so the pre-selection vector
; coalescer does not see both halves of a single vector.
define void @unpack_vector_bitcasts(i32 %word) {
; CHECK-LABEL: unpack_vector_bitcasts(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK-NOT: mov.b32 {
; CHECK: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK-NOT: mov.b32 {
; CHECK: // halves [[LO]], [[HI]]
  %vf = bitcast i32 %word to <2 x half>
  %vb = bitcast i32 %word to <2 x bfloat>
  %lo = extractelement <2 x half> %vf, i32 0
  %hi = extractelement <2 x bfloat> %vb, i32 1
  call void asm sideeffect "// halves $0, $1", "h,h"(half %lo, bfloat %hi)
  ret void
}

; Also merge vector extracts with scalar truncations, in either direction.
define void @unpack_scalar_low_vector_high(i32 %word) {
; CHECK-LABEL: unpack_scalar_low_vector_high(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK-NOT: cvt.
; CHECK: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK-NOT: mov.b32 {
; CHECK: // halves [[LO]], [[HI]]
  %lo = trunc i32 %word to i16
  %v = bitcast i32 %word to <2 x half>
  %hi = extractelement <2 x half> %v, i32 1
  call void asm sideeffect "// halves $0, $1", "h,h"(i16 %lo, half %hi)
  ret void
}

define void @unpack_vector_low_scalar_high(i32 %word) {
; CHECK-LABEL: unpack_vector_low_scalar_high(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK-NOT: mov.b32 {
; CHECK: // halves [[HI]], [[LO]]
  %v = bitcast i32 %word to <2 x bfloat>
  %lo = extractelement <2 x bfloat> %v, i32 0
  %shift = lshr i32 %word, 16
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// halves $0, $1", "h,h"(i16 %hi, bfloat %lo)
  ret void
}

; Duplicate low-half extracts with different types must share the same result.
define void @unpack_vector_multiuse(i32 %word) {
; CHECK-LABEL: unpack_vector_multiuse(
; CHECK: ld.param.b32 [[WORD:%r[0-9]+]],
; CHECK: mov.b32 {[[LO:%rs[0-9]+]], [[HI:%rs[0-9]+]]}, [[WORD]];
; CHECK-NOT: mov.b32 {
; CHECK-NOT: cvt.
; CHECK: // halves [[LO]], [[HI]], [[LO]], [[WORD]], [[HI]]
  %v = bitcast i32 %word to <2 x half>
  %lo = extractelement <2 x half> %v, i32 0
  %lo_int = trunc i32 %word to i16
  %shift = lshr i32 %word, 16
  %hi = trunc i32 %shift to i16
  call void asm sideeffect "// halves $0, $1, $2, $3, $4", "h,h,h,r,h"(half %lo, i16 %hi, i16 %lo_int, i32 %word, i16 %hi)
  ret void
}

; Distinct results of one DAG node are different source words.
define void @unpack_vector_different_results() {
; CHECK-LABEL: unpack_vector_different_results(
; CHECK: mov.b32 [[A:%r[0-9]+]], 1; mov.b32 [[B:%r[0-9]+]], 2;
; OLD-DAG: { .reg .b16 tmp; mov.b32 {[[LO:%rs[0-9]+]], tmp}, [[A]]; }
; OLD-DAG: { .reg .b16 tmp; mov.b32 {tmp, [[HI:%rs[0-9]+]]}, [[B]]; }
; NEW-DAG: mov.b32 {[[LO:%rs[0-9]+]], _}, [[A]];
; NEW-DAG: mov.b32 {_, [[HI:%rs[0-9]+]]}, [[B]];
; CHECK: // halves [[LO]], [[HI]]
  %words = call {i32, i32} asm "mov.b32 $0, 1; mov.b32 $1, 2;", "=r,=r"()
  %a = extractvalue {i32, i32} %words, 0
  %b = extractvalue {i32, i32} %words, 1
  %va = bitcast i32 %a to <2 x half>
  %vb = bitcast i32 %b to <2 x half>
  %lo = extractelement <2 x half> %va, i32 0
  %hi = extractelement <2 x half> %vb, i32 1
  call void asm sideeffect "// halves $0, $1", "h,h"(half %lo, half %hi)
  ret void
}
