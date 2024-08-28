; RUN: llc --mcpu=sapphirerapids -mtriple=x86_64 -denormal-fp-math=preserve-sign < %s | FileCheck %s
; RUN: llc --mcpu=sapphirerapids -mtriple=x86_64 -denormal-fp-math=ieee < %s | FileCheck -check-prefix=IEEE-DENORMAL %s 
; RUN: llc --mcpu=sapphirerapids -mtriple=x86_64 -denormal-fp-math=ieee < %s | FileCheck -check-prefix=DYN-DENORMAL %s

define void @canonicalize_denormal1_f32_pre_sign(float addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI0_0:
  ; CHECK: 	.long	0x80000000                      # float -0
  ; CHECK-LABEL: canonicalize_denormal1_f32_pre_sign:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovss	.LCPI0_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovss	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 2155872255 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_f64_pre_sign(double addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI1_0:
  ; CHECK: 	.quad	0x8000000000000000              # double -0
  ; CHECK-LABEL: canonicalize_denormal1_f64_pre_sign:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovsd	.LCPI1_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 9227875636482146303 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}


define void @canonicalize_qnan_f64(double addrspace(1)* %out) {
  ;cCHECK-LABEL: .LCPI2_0:
  ;cCHECK: 	.quad	0x7ff8000000000000              # double NaN
  ; CHECK-LABEL: canonicalize_qnan_f64:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovsd	.LCPI2_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call double @llvm.canonicalize.f64(double 0x7FF8000000000000)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

define void @canonicalize_qnan_value_neg1_f64(double addrspace(1)* %out) {
  ;cCHECK-LABEL: .LCPI3_0:
  ;cCHECK: 	.quad	0xffffffffffffffff              # double NaN
  ; CHECK-LABEL: canonicalize_qnan_value_neg1_f64:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovsd	.LCPI3_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 -1 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

define void @canonicalize_qnan_value_neg2_f64(double addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI4_0:
  ; CHECK: 	.quad	0xfffffffffffffffe              # double NaN
  ; CHECK-LABEL: canonicalize_qnan_value_neg2_f64:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovsd	.LCPI4_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 -2 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

define void @canonicalize_snan0_value_f64(double addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI5_0:
  ; CHECK: 	.quad	0x7ff8000000000000              # double NaN
  ; CHECK-LABEL: canonicalize_snan0_value_f64:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovsd	.LCPI5_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 9218868437227405313 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

define void @canonicalize_undef(double addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI6_0:
  ; CHECK: 	.quad	0x7ff8000000000000              # double NaN
  ; CHECK-LABEL: canonicalize_undef:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovsd	.LCPI6_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call double @llvm.canonicalize.f64(double undef)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_f32_ieee(float addrspace(1)* %out) {
  ; IEEE-DENORMAL-LABEL: .LCPI7_0:
  ; IEEE-DENORMAL: 	.long	0x807fffff                      # float -1.17549421E-38
  ; IEEE-DENORMAL-LABEL: canonicalize_denormal1_f32_ieee:
  ; IEEE-DENORMAL: # %bb.0:
  ; IEEE-DENORMAL-NEXT: 	vmovss	.LCPI7_0(%rip), %xmm0
  ; IEEE-DENORMAL-NEXT: 	vmovss	%xmm0, (%rdi)
  ; IEEE-DENORMAL-NEXT: 	retq

  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 2155872255 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_f64_ieee(double addrspace(1)* %out) {
  ; IEEE-DENORMAL-LABEL: .LCPI8_0:
  ; IEEE-DENORMAL: 	.quad	0x800fffffffffffff              # double -2.2250738585072009E-308
  ; IEEE-DENORMAL-LABEL: canonicalize_denormal1_f64_ieee:
  ; IEEE-DENORMAL: # %bb.0:
  ; IEEE-DENORMAL-NEXT: 	vmovsd	.LCPI8_0(%rip), %xmm0
  ; IEEE-DENORMAL-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; IEEE-DENORMAL-NEXT: 	retq

  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 9227875636482146303 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_f32_dynamic(float addrspace(1)* %out) {
  ; DYN-DENORMAL-LABEL: .LCPI9_0:
  ; DYN-DENORMAL: 	.long	0x807fffff                      # float -1.17549421E-38
  ; DYN-DENORMAL-LABEL: canonicalize_denormal1_f32_dynamic:
  ; DYN-DENORMAL: # %bb.0:
  ; DYN-DENORMAL-NEXT: 	vmovss	.LCPI9_0(%rip), %xmm0
  ; DYN-DENORMAL-NEXT: 	vmovss	%xmm0, (%rdi)
  ; DYN-DENORMAL-NEXT: 	retq

  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 2155872255 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_f64_dynamic(double addrspace(1)* %out) {
  ; DYN-DENORMAL-LABEL: .LCPI10_0:
  ; DYN-DENORMAL: 	.quad	0x800fffffffffffff              # double -2.2250738585072009E-308
  ; DYN-DENORMAL-LABEL: canonicalize_denormal1_f64_dynamic:
  ; DYN-DENORMAL: # %bb.0:
  ; DYN-DENORMAL-NEXT: 	vmovsd	.LCPI10_0(%rip), %xmm0
  ; DYN-DENORMAL-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; DYN-DENORMAL-NEXT: 	retq

  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 9227875636482146303 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_bfloat_pre_sign(bfloat addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI11_0:
  ; CHECK: 	.long	0x80000000                      # float -0
  ; CHECK-LABEL: canonicalize_denormal1_bfloat_pre_sign:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovss	.LCPI11_0(%rip), %xmm0
  ; CHECK-NEXT: 	vcvtneps2bf16	%xmm0, %xmm0
  ; CHECK-NEXT: 	vpextrw	$0, %xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call bfloat @llvm.canonicalize.bf16(bfloat bitcast (i16 32768 to bfloat))
  store bfloat %canonicalized, bfloat addrspace(1)* %out
  ret void
}


define void @canonicalize_denormal1_bfloat_ieee(bfloat addrspace(1)* %out) {
  ; IEEE-DENORMAL-LABEL: .LCPI12_0:
  ; IEEE-DENORMAL: 	.long	0x80000000                      # float -0
  ; IEEE-DENORMAL-LABEL: canonicalize_denormal1_bfloat_ieee:
  ; IEEE-DENORMAL: # %bb.0:
  ; IEEE-DENORMAL-NEXT: 	vmovss	.LCPI12_0(%rip), %xmm0
  ; IEEE-DENORMAL-NEXT: 	vcvtneps2bf16	%xmm0, %xmm0
  ; IEEE-DENORMAL-NEXT: 	vpextrw	$0, %xmm0, (%rdi)
  ; IEEE-DENORMAL-NEXT: 	retq

  %canonicalized = call bfloat @llvm.canonicalize.bf16(bfloat bitcast (i16 32768 to bfloat))
  store bfloat %canonicalized, bfloat addrspace(1)* %out
  ret void
}


define void @canonicalize_denormal1_bfloat_dynamic(bfloat addrspace(1)* %out) {
  ; DYN-DENORMAL-LABEL: .LCPI13_0:
  ; DYN-DENORMAL: 	.long	0x80000000                      # float -0
  ; DYN-DENORMAL-LABEL: canonicalize_denormal1_bfloat_dynamic:
  ; DYN-DENORMAL: # %bb.0:
  ; DYN-DENORMAL-NEXT: 	vmovss	.LCPI13_0(%rip), %xmm0
  ; DYN-DENORMAL-NEXT: 	vcvtneps2bf16	%xmm0, %xmm0
  ; DYN-DENORMAL-NEXT: 	vpextrw	$0, %xmm0, (%rdi)
  ; DYN-DENORMAL-NEXT: 	retq

  %canonicalized = call bfloat @llvm.canonicalize.bf16(bfloat bitcast (i16 32768 to bfloat))
  store bfloat %canonicalized, bfloat addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_half_pre_sign(half addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI14_0:
  ; CHECK: 	.short	0x8000                      # half -0
  ; CHECK-LABEL: canonicalize_denormal1_half_pre_sign:
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	vmovsh	.LCPI14_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsh	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 32768 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}


define void @canonicalize_denormal1_half_ieee(half addrspace(1)* %out) {
  ; IEEE-DENORMAL-LABEL: .LCPI15_0:
  ; IEEE-DENORMAL: 	.short	0x8000                          # half -0
  ; IEEE-DENORMAL-LABEL: canonicalize_denormal1_half_ieee:
  ; IEEE-DENORMAL: # %bb.0:
  ; IEEE-DENORMAL-NEXT: 	vmovsh	.LCPI15_0(%rip), %xmm0
  ; IEEE-DENORMAL-NEXT: 	vmovsh	%xmm0, (%rdi)
  ; IEEE-DENORMAL-NEXT: 	retq

  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 32768 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_half_dynamic(half addrspace(1)* %out) {
  ; DYN-DENORMAL-LABEL: .LCPI16_0:
  ; DYN-DENORMAL: 	.short	0x8000                          # half -0
  ; DYN-DENORMAL-LABEL: canonicalize_denormal1_half_dynamic:
  ; DYN-DENORMAL: # %bb.0:
  ; DYN-DENORMAL-NEXT: 	vmovsh	.LCPI16_0(%rip), %xmm0
  ; DYN-DENORMAL-NEXT: 	vmovsh	%xmm0, (%rdi)
  ; DYN-DENORMAL-NEXT: 	retq

  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 32768 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_x86_fp80_pre_sign(x86_fp80 addrspace(1)* %out) {
    ; CHECK-LAEBL: .LCPI17_0:
    ; CHECK: .long	0x00000000                      # float 0
    ; CHECK-LAEBL: canonicalize_denormal1_x86_fp80_pre_sign
    ; CHECK: # %bb.0:
    ; CHECK-NEXT: flds	.LCPI17_0(%rip)
    ; CHECK-NEXT: fstpt	(%rdi)
    ; CHECK-NEXT: retq
    %canonicalized = call x86_fp80 @llvm.canonicalize.f80(x86_fp80 0xK00000000000000000001)
    store x86_fp80 %canonicalized, x86_fp80 addrspace(1)* %out
    ret void
}

define void @canonicalize_denormal1_x86_fp80_dynamic(x86_fp80 addrspace(1)* %out) {
  ; DYN-DENORMAL-LAEBL: .LCPI17_0:
  ; DYN-DENORMAL: .quad	0x0000000000000001              # x86_fp80 3.64519953188247460253E-4951
  ; DYN-DENORMAL-LAEBL: canonicalize_denormal1_x86_fp80_dynamic
  ; DYN-DENORMAL: # %bb.0:
  ; DYN-DENORMAL-NEXT: fldt	.LCPI17_0(%rip)
  ; DYN-DENORMAL-NEXT: fstpt	(%rdi)
  ; DYN-DENORMAL-NEXT: retq
  %canonicalized = call x86_fp80 @llvm.canonicalize.f80(x86_fp80 0xK00000000000000000001)
  store x86_fp80 %canonicalized, x86_fp80 addrspace(1)* %out
  ret void
}

define void @canonicalize_denormal1_x86_fp80_ieee(x86_fp80 addrspace(1)* %out) {
  ; IEEE-DENORMAL-LAEBL: .LCPI17_0:
  ; IEEE-DENORMAL: .quad	0x0000000000000001              # x86_fp80 3.64519953188247460253E-4951
  ; IEEE-DENORMAL-LAEBL: canonicalize_denormal1_x86_fp80_ieee
  ; IEEE-DENORMAL: # %bb.0:
  ; IEEE-DENORMAL-NEXT: fldt	.LCPI17_0(%rip)
  ; IEEE-DENORMAL-NEXT: fstpt	(%rdi)
  ; IEEE-DENORMAL-NEXT: retq
  %canonicalized = call x86_fp80 @llvm.canonicalize.f80(x86_fp80 0xK00000000000000000001)
  store x86_fp80 %canonicalized, x86_fp80 addrspace(1)* %out
  ret void
}