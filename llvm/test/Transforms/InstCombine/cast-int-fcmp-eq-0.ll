; RUN: opt -S -instcombine < %s | FileCheck %s

; CHECK-LABEL: @i32_cast_cmp_oeq_int_0_uitofp(
; CHECK-NEXT: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_n0_uitofp(
; CHECK-NEXT: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_n0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_0_sitofp(
; CHECK-NEXT: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_n0_sitofp(
; CHECK-NEXT: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_n0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_one_int_0_uitofp(
; CHECK-NEXT: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_one_int_0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp one float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_one_int_n0_uitofp(
; CHECK-NEXT: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_one_int_n0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp one float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_one_int_0_sitofp(
; CHECK-NEXT: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_one_int_0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp one float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_one_int_n0_sitofp(
; CHECK-NEXT: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_one_int_n0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp one float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ueq_int_0_uitofp(
; CHECK-NEXT: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ueq_int_0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp ueq float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ueq_int_n0_uitofp(
; CHECK-NEXT: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ueq_int_n0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp ueq float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ueq_int_0_sitofp(
; CHECK-NEXT: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ueq_int_0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp ueq float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ueq_int_n0_sitofp(
; CHECK-NEXT: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ueq_int_n0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp ueq float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_une_int_0_uitofp(
; CHECK-NEXT: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_une_int_0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp une float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_une_int_n0_uitofp(
; CHECK-NEXT: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_une_int_n0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp une float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_une_int_0_sitofp(
; CHECK-NEXT: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_une_int_0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp une float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_une_int_n0_sitofp(
; CHECK-NEXT: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_une_int_n0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp une float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ogt_int_0_uitofp(
; CHECK: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ogt_int_0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp ogt float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ogt_int_n0_uitofp(
; CHECK: icmp ne i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ogt_int_n0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp ogt float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ogt_int_0_sitofp(
; CHECK: icmp sgt i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ogt_int_0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp ogt float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ogt_int_n0_sitofp(
; CHECK: icmp sgt i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ogt_int_n0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp ogt float %f, -0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ole_int_0_uitofp(
; CHECK: icmp eq i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ole_int_0_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp ole float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ole_int_0_sitofp(
; CHECK: icmp slt i32 %i, 1
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_ole_int_0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp ole float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_olt_int_0_sitofp(
; CHECK: icmp slt i32 %i, 0
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_olt_int_0_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp olt float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i64_cast_cmp_oeq_int_0_uitofp(
; CHECK-NEXT: icmp eq i64 %i, 0
; CHECK-NEXT: ret
define i1 @i64_cast_cmp_oeq_int_0_uitofp(i64 %i) {
  %f = uitofp i64 %i to float
  %cmp = fcmp oeq float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i64_cast_cmp_oeq_int_0_sitofp(
; CHECK-NEXT: icmp eq i64 %i, 0
; CHECK-NEXT: ret
define i1 @i64_cast_cmp_oeq_int_0_sitofp(i64 %i) {
  %f = sitofp i64 %i to float
  %cmp = fcmp oeq float %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i64_cast_cmp_oeq_int_0_uitofp_half(
; CHECK-NEXT: icmp eq i64 %i, 0
; CHECK-NEXT: ret
define i1 @i64_cast_cmp_oeq_int_0_uitofp_half(i64 %i) {
  %f = uitofp i64 %i to half
  %cmp = fcmp oeq half %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i64_cast_cmp_oeq_int_0_sitofp_half(
; CHECK-NEXT: icmp eq i64 %i, 0
; CHECK-NEXT: ret
define i1 @i64_cast_cmp_oeq_int_0_sitofp_half(i64 %i) {
  %f = sitofp i64 %i to half
  %cmp = fcmp oeq half %f, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_0_uitofp_ppcf128(
; CHECK: uitofp
; CHECK: fcmp oeq
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_0_uitofp_ppcf128(i32 %i) {
  %f = uitofp i32 %i to ppc_fp128
  %cmp = fcmp oeq ppc_fp128 %f, 0xM00000000000000000000000000000000
  ret i1 %cmp
}

; Since 0xFFFFFF fits in a float, and one less and 
; one more than it also fits without rounding, the 
; test can be optimized to an integer compare.

; CHECK-LABEL: @i32_cast_cmp_oeq_int_i24max_uitofp(
; CHECK: icmp eq i32 %i, 16777215
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_i24max_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x416FFFFFE0000000
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_i24max_sitofp(
; CHECK: icmp eq i32 %i, 16777215
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_i24max_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x416FFFFFE0000000
  ret i1 %cmp
}

; Though 0x1000000 fits in a float, one more than it 
; would round to it too, hence a single integer comparison 
; does not suffice.

; CHECK-LABEL: @i32_cast_cmp_oeq_int_i24maxp1_uitofp(
; CHECK: uitofp
; CHECK: fcmp oeq

; XCHECK: icmp eq i32 %i, 16777216
; XCHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_i24maxp1_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x4170000000000000
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_i24maxp1_sitofp(
; CHECK: sitofp
; CHECK: fcmp oeq

; XCHECK: icmp eq i32 %i, 16777216
; XCHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_i24maxp1_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x4170000000000000
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_i32umax_uitofp(
; CHECK: uitofp
; CHECK: fcmp oeq
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_i32umax_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x41F0000000000000
  ret i1 %cmp
}

; 32-bit unsigned integer cannot possibly round up to 1<<33
; CHECK-LABEL: @i32_cast_cmp_oeq_int_big_uitofp(
; CHECK-NEXT: ret i1 false
define i1 @i32_cast_cmp_oeq_int_big_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x4200000000000000
  ret i1 %cmp
}

; 32-bit signed integer cannot possibly round up to 1<<32
; CHECK-LABEL: @i32_cast_cmp_oeq_int_i32umax_sitofp(
; CHECK-NEXT: ret i1 false
define i1 @i32_cast_cmp_oeq_int_i32umax_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x41F0000000000000
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_i32imin_sitofp(
; CHECK: sitofp
; CHECK: fcmp oeq
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_i32imin_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0xC1E0000000000000
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_i32imax_uitofp(
; CHECK: uitofp
; CHECK: fcmp oeq
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_i32imax_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x41E0000000000000
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_i32imax_sitofp(
; CHECK: sitofp
; CHECK: fcmp oeq
; CHECK-NEXT: ret
define i1 @i32_cast_cmp_oeq_int_i32imax_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x41E0000000000000
  ret i1 %cmp
}

; 32-bit signed integer cannot possibly round to -1<<32
; CHECK-LABEL: @i32_cast_cmp_oeq_int_negi32umax_sitofp(
; CHECK-NEXT: ret i1 false
define i1 @i32_cast_cmp_oeq_int_negi32umax_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0xC1F0000000000000
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_half_uitofp(
; CHECK: ret i1 false
define i1 @i32_cast_cmp_oeq_half_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0.5
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_half_sitofp(
; CHECK: ret i1 false
define i1 @i32_cast_cmp_oeq_half_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0.5
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_one_half_uitofp(
; CHECK: ret i1 true
define i1 @i32_cast_cmp_one_half_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp one float %f, 0.5
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_one_half_sitofp(
; CHECK: ret i1 true
define i1 @i32_cast_cmp_one_half_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp one float %f, 0.5
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ueq_half_uitofp(
; CHECK: ret i1 false
define i1 @i32_cast_cmp_ueq_half_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp ueq float %f, 0.5
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_ueq_half_sitofp(
; CHECK: ret i1 false
define i1 @i32_cast_cmp_ueq_half_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp ueq float %f, 0.5
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_une_half_uitofp(
; CHECK: ret i1 true
define i1 @i32_cast_cmp_une_half_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp une float %f, 0.5
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_une_half_sitofp(
; CHECK: ret i1 true
define i1 @i32_cast_cmp_une_half_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp une float %f, 0.5
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_inf_uitofp(
; CHECK-NEXT: ret i1 false
define i1 @i32_cast_cmp_oeq_int_inf_uitofp(i32 %i) {
  %f = uitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x7FF0000000000000
  ret i1 %cmp
}

; CHECK-LABEL: @i32_cast_cmp_oeq_int_inf_sitofp(
; CHECK-NEXT: ret i1 false
define i1 @i32_cast_cmp_oeq_int_inf_sitofp(i32 %i) {
  %f = sitofp i32 %i to float
  %cmp = fcmp oeq float %f, 0x7FF0000000000000
  ret i1 %cmp
}

; An i128 could round to an IEEE single-precision infinity.
; CHECK-LABEL: @i128_cast_cmp_oeq_int_inf_uitofp(
; CHECK: uitofp
; CHECK: fcmp oeq
; CHECK-NEXT: ret
define i1 @i128_cast_cmp_oeq_int_inf_uitofp(i128 %i) {
  %f = uitofp i128 %i to float
  %cmp = fcmp oeq float %f, 0x7FF0000000000000
  ret i1 %cmp
}
