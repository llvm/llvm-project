; RUN: llc --mcpu=sapphirerapids -mtriple=x86_64 < %s | FileCheck %s

define float @canon_fp32() {
  ; CHECK-LABEL: .LCPI0_0:
  ; CHECK: .long	0x40400000                      # float 3
  ; CHECK-LABEL: canon_fp32
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: vmovss	.LCPI0_0(%rip), %xmm0           # xmm0 = [3.0E+0,0.0E+0,0.0E+0,0.0E+0]
  ; CHECK-NEXT: retq
  %canonicalized = call float @llvm.canonicalize.f32(float 3.0)
  ret float %canonicalized
}

define half @canon_fp16() {
  ; CHECK-LABEL: .LCPI1_0:
  ; CHECK: .short	0x4200                          # half 3
  ; CHECK-LABEL: canon_fp16
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: vmovsh	.LCPI1_0(%rip), %xmm0
  ; CHECK-NEXT: retq
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH4200) ; half 3.0
  ret half %canonicalized
}

define double @canon_fp64() {
  ; CHECK-LABEL: .LCPI2_0:
  ; CHECK: .quad	0x4008000000000000              # double 3
  ; CHECK-LABEL: canon_fp64
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: vmovsd	.LCPI2_0(%rip), %xmm0
  ; CHECK-NEXT: retq
  %canonicalized = call double @llvm.canonicalize.f64(double 3.0)
  ret double %canonicalized
}

define x86_fp80 @canon_fp80() {
  ; CHECK-LABEL: .LCPI3_0:
  ; CHECK: .long	0x42b40000                      # float 90
  ; CHECK-LABEL: canon_fp80
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: flds	.LCPI3_0(%rip)
  ; CHECK-NEXT: retq

  %canonicalized = call x86_fp80 @llvm.canonicalize.f80(x86_fp80 0xK4005B400000000000000) ; 90.0
  ret x86_fp80 %canonicalized
}


define x86_fp80 @complex_canonicalize_x86_fp80(x86_fp80 %a, x86_fp80 %b) {
entry:
  ; CHECK-LABEL: .LCPI4_0:
  ; CHECK: .long	0x42b40000                      # float 90
  ; CHECK-LABEL: complex_canonicalize_x86_fp80
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: 	fldt	24(%rsp)
  ; CHECK-NEXT: 	flds	.LCPI4_0(%rip)
  ; CHECK-NEXT: 	fsubp	%st, %st(1)
  ; CHECK-NEXT: 	retq

  %mul1 = fsub x86_fp80 %a, %b
  %add = fadd x86_fp80 %mul1, %b
  %mul2 = fsub x86_fp80 %add, %mul1
  %canonicalized = call x86_fp80 @llvm.canonicalize.f80(x86_fp80 0xK4005B400000000000000)
  %result = fsub x86_fp80 %canonicalized, %b
  ret x86_fp80 %result
}

define double @complex_canonicalize_fp64(double %a, double %b) unnamed_addr #0 {
start:
  ; CHECK-LABEL: .LCPI5_0:
  ; CHECK: .quad	0x4008000000000000              # double 3
  ; CHECK-LABEL: complex_canonicalize_fp64
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: vmovsd	.LCPI5_0(%rip), %xmm0
  ; CHECK-NEXT: retq

  %c = fcmp olt double %a, %b
  %d = fcmp uno double %a, 0.000000e+00
  %or.cond.i.i = or i1 %d, %c
  %e = select i1 %or.cond.i.i, double %b, double %a
  %f = tail call double @llvm.canonicalize.f64(double 3.0) #2
  ret double %f
}

define void @test_fold_canonicalize_p0_f32(float addrspace(1)* %out) #1 {
  ; CHECK-LAEBL: test_fold_canonicalize_p0_f32
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: vxorps	%xmm0, %xmm0, %xmm0
  ; CHECK-NEXT: vmovss	%xmm0, (%rdi)
  ; CHECK-NEXT: retq
  %canonicalized = call float @llvm.canonicalize.f32(float 0.0)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

define void @test_fold_canonicalize_n0_f32(float addrspace(1)* %out) #1 {
  ; CHECK-LAEBL: .LCPI7_0:
  ; CHECK: .long	0x80000000                      # float -0
  ; CHECK-LAEBL: test_fold_canonicalize_n0_f32
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: vmovss	.LCPI7_0(%rip), %xmm0
  ; CHECK-NEXT: vmovss	%xmm0, (%rdi)
  ; CHECK-NEXT: retq
  %canonicalized = call float @llvm.canonicalize.f32(float -0.0)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}


define void @v_test_canonicalize_p90_x86_fp80(x86_fp80 addrspace(1)* %out) #1 {
  ; CHECK-LAEBL: .LCPI8_0:
  ; CHECK: .long	0x42b40000                      # float 90
  ; CHECK-LAEBL: v_test_canonicalize_p90_x86_fp80
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: flds	.LCPI8_0(%rip)
  ; CHECK-NEXT: fstpt	(%rdi)
  ; CHECK-NEXT: retq
  %canonicalized = call x86_fp80 @llvm.canonicalize.f80(x86_fp80 0xK4005B400000000000000)
  store x86_fp80 %canonicalized, x86_fp80 addrspace(1)* %out
  ret void
}

define void @v_test_canonicalize_p3__half(half addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI9_0:
  ; CHECK: .short	0x4200                          # half 3
  ; CHECK-LABEL: v_test_canonicalize_p3__half:
  ; CHECK:       # %bb.0: # %entry
  ; CHECK-NEXT:   vmovsh	.LCPI9_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsh	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

entry:
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH4200)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

define void @v_test_canonicalize_p3_f64(double addrspace(1)* %out) #1 {
  ; CHECK-LABEL: .LCPI10_0:
  ; CHECK: .quad	0x4008000000000000              # double 3
  ; CHECK-LAEBL: v_test_canonicalize_p3_f64
  ; CHECK: # %bb.0:
  ; CHECK-NEXT:       vmovsd	.LCPI10_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsd	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq
entry:
  %canonicalized = call double @llvm.canonicalize.f64(double 3.0)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

define void @v_test_canonicalize_p3__bfloat(bfloat addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI11_0:
  ; CHECK: .long	0x40400000                      # float 3
  ; CHECK-LABEL: v_test_canonicalize_p3__bfloat:
  ; CHECK:       # %bb.0: # %entry
  ; CHECK-NEXT: 	vmovss	.LCPI11_0(%rip), %xmm0          # xmm0 = [3.0E+0,0.0E+0,0.0E+0,0.0E+0]
  ; CHECK-NEXT: 	vcvtneps2bf16	%xmm0, %xmm0
  ; CHECK-NEXT: 	vpextrw	$0, %xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

entry:
  %canonicalized = call bfloat @llvm.canonicalize.bf16(bfloat 3.0)
  store bfloat %canonicalized, bfloat addrspace(1)* %out
  ret void
}

define void @v_test_canonicalize_n3__bfloat(bfloat addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI12_0:
  ; CHECK: .long	0xc0400000                      # float -3
  ; CHECK-LABEL: v_test_canonicalize_n3__bfloat:
  ; CHECK:       # %bb.0: # %entry
  ; CHECK-NEXT: 	vmovss	.LCPI12_0(%rip), %xmm0          # xmm0 = [-3.0E+0,0.0E+0,0.0E+0,0.0E+0]
  ; CHECK-NEXT: 	vcvtneps2bf16	%xmm0, %xmm0
  ; CHECK-NEXT: 	vpextrw	$0, %xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

entry:
  %canonicalized = call bfloat @llvm.canonicalize.bf16(bfloat -3.0)
  store bfloat %canonicalized, bfloat addrspace(1)* %out
  ret void
}

define void @v_test_canonicalize_n90_x86_fp80(x86_fp80 addrspace(1)* %out) #1 {
  ; CHECK-LAEBL: .LCPI13_0:
  ; CHECK: .long	0xc2b40000                      # float -90
  ; CHECK-LAEBL: v_test_canonicalize_n90_x86_fp80
  ; CHECK: # %bb.0:
  ; CHECK-NEXT: flds	.LCPI13_0(%rip)
  ; CHECK-NEXT: fstpt	(%rdi)
  ; CHECK-NEXT: retq
  %canonicalized = call x86_fp80 @llvm.canonicalize.f80(x86_fp80 0xKC005B400000000000000)
  store x86_fp80 %canonicalized, x86_fp80 addrspace(1)* %out
  ret void
}

define void @v_test_canonicalize_n3__half(half addrspace(1)* %out) {
  ; CHECK-LABEL: .LCPI14_0:
  ; CHECK: .short	0xc200                          # half -3
  ; CHECK-LABEL: v_test_canonicalize_n3__half:
  ; CHECK:       # %bb.0: # %entry
  ; CHECK-NEXT:   vmovsh	.LCPI14_0(%rip), %xmm0
  ; CHECK-NEXT: 	vmovsh	%xmm0, (%rdi)
  ; CHECK-NEXT: 	retq

entry:
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHC200)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}