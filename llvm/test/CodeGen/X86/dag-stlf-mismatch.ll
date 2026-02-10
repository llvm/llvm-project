; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-windows-gnu -mcpu=tigerlake | FileCheck %s --check-prefix=TIGER

%struct.Data = type { float }

define float @test_stlf_integer(ptr %p, float %v) {
; CHECK-LABEL: test_stlf_integer:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl $0, (%rdi)
; CHECK-NEXT:    xorps %xmm1, %xmm1
; CHECK-NEXT:    mulss %xmm1, %xmm0
; CHECK-NEXT:    retq
  store i32 0, ptr %p, align 4
  %f = load float, ptr %p, align 4
  %r = fmul float %f, %v
  ret float %r
}

define float @test_stlf_vector(ptr %p, float %v) {
; CHECK-LABEL: test_stlf_vector:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm1, %xmm1
; CHECK-NEXT:    movups %xmm1, (%rdi)
; CHECK-NEXT:    mulss (%rdi), %xmm0
; CHECK-NEXT:    retq
  store <4 x float> zeroinitializer, ptr %p, align 4
  %f = load float, ptr %p, align 4
  %r = fmul float %f, %v
  ret float %r
}

define float @test_stlf_bitcast(ptr %p, float %v) {
; CHECK-LABEL: test_stlf_bitcast:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm1, %xmm1
; CHECK-NEXT:    movups %xmm1, (%rdi)
; CHECK-NEXT:    mulss (%rdi), %xmm0
; CHECK-NEXT:    retq
  store <2 x i64> zeroinitializer, ptr %p, align 4
  %f = load float, ptr %p, align 4
  %r = fmul float %f, %v
  ret float %r
}

declare void @ext_func(ptr byval(%struct.Data) align 4 %p)
define void @test_stlf_late_byval(ptr %ptr) nounwind {
; CHECK-LABEL: test_stlf_late_byval:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    movl $0, (%rdi)
; CHECK-NEXT:    movl $0, (%rsp)
; CHECK-NEXT:    callq ext_func@PLT
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    retq
  store i32 0, ptr %ptr, align 4
  call void @ext_func(ptr byval(%struct.Data) align 4 %ptr)
  ret void
}

define float @test_stlf_variable(ptr %p, i32 %val, float %v) {
; CHECK-LABEL: test_stlf_variable:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movd %esi, %xmm1
; CHECK-NEXT:    movl %esi, (%rdi)
; CHECK-NEXT:    mulss %xmm1, %xmm0
; CHECK-NEXT:    retq
  store i32 %val, ptr %p, align 4
  %f = load float, ptr %p, align 4
  %r = fmul float %f, %v
  ret float %r
}

define <32 x i1> @v32i1_bitcast_crash(<4 x i8> %arg) nounwind {
; TIGER-LABEL: v32i1_bitcast_crash:
; TIGER:       # %bb.0:
; TIGER-NEXT:    subq $24, %rsp
; TIGER-NEXT:    vmovaps (%rcx), %xmm0
; TIGER-NEXT:    vmovaps %xmm0, (%rsp)
; TIGER-NEXT:    kmovd (%rsp), %k0
; TIGER-NEXT:    vpmovm2b %k0, %ymm0
; TIGER-NEXT:    addq $24, %rsp
; TIGER-NEXT:    retq
  %res = bitcast <4 x i8> %arg to <32 x i1>
  ret <32 x i1> %res
}
