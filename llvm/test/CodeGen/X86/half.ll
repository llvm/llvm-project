; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=-f16c -asm-verbose=false \
; RUN:   | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LIBCALL
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+f16c -asm-verbose=false \
; RUN:    | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-F16C

define void @test_load_store(half* %in, half* %out) {
; CHECK-LABEL: test_load_store:
; CHECK: movw (%rdi), [[TMP:%[a-z0-9]+]]
; CHECK: movw [[TMP]], (%rsi)
  %val = load half, half* %in
  store half %val, half* %out
  ret void
}

define i16 @test_bitcast_from_half(half* %addr) {
; CHECK-LABEL: test_bitcast_from_half:
; CHECK: movzwl (%rdi), %eax
  %val = load half, half* %addr
  %val_int = bitcast half %val to i16
  ret i16 %val_int
}

define void @test_bitcast_to_half(half* %addr, i16 %in) {
; CHECK-LABEL: test_bitcast_to_half:
; CHECK: movw %si, (%rdi)
  %val_fp = bitcast i16 %in to half
  store half %val_fp, half* %addr
  ret void
}

define float @test_extend32(half* %addr) {
; CHECK-LABEL: test_extend32:

; CHECK-LIBCALL: jmp __gnu_h2f_ieee
; CHECK-F16C: vcvtph2ps
  %val16 = load half, half* %addr
  %val32 = fpext half %val16 to float
  ret float %val32
}

define double @test_extend64(half* %addr) {
; CHECK-LABEL: test_extend64:

; CHECK-LIBCALL: callq __gnu_h2f_ieee
; CHECK-LIBCALL: cvtss2sd
; CHECK-F16C: vcvtph2ps
; CHECK-F16C: vcvtss2sd
  %val16 = load half, half* %addr
  %val32 = fpext half %val16 to double
  ret double %val32
}

define void @test_trunc32(float %in, half* %addr) {
; CHECK-LABEL: test_trunc32:

; CHECK-LIBCALL: callq __gnu_f2h_ieee
; CHECK-F16C: vcvtps2ph
  %val16 = fptrunc float %in to half
  store half %val16, half* %addr
  ret void
}

define void @test_trunc64(double %in, half* %addr) {
; CHECK-LABEL: test_trunc64:

; CHECK-LIBCALL: callq __truncdfhf2
; CHECK-F16C: callq __truncdfhf2
  %val16 = fptrunc double %in to half
  store half %val16, half* %addr
  ret void
}

define i64 @test_fptosi_i64(half* %p) #0 {
; CHECK-LABEL: test_fptosi_i64:

; CHECK-LIBCALL-NEXT: pushq %rax
; CHECK-LIBCALL-NEXT: movzwl (%rdi), %edi
; CHECK-LIBCALL-NEXT: callq __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: cvttss2si %xmm0, %rax
; CHECK-LIBCALL-NEXT: popq %rdx
; CHECK-LIBCALL-NEXT: retq

; CHECK-F16C-NEXT: movswl (%rdi), [[REG0:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vmovd [[REG0]], [[REG1:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vcvtph2ps [[REG1]], [[REG2:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vcvttss2si [[REG2]], %rax
; CHECK-F16C-NEXT: retq
  %a = load half, half* %p, align 2
  %r = fptosi half %a to i64
  ret i64 %r
}

define void @test_sitofp_i64(i64 %a, half* %p) #0 {
; CHECK-LABEL: test_sitofp_i64:

; CHECK-LIBCALL-NEXT: pushq [[ADDR:%[a-z]+]]
; CHECK-LIBCALL-NEXT: movq %rsi, [[ADDR]]
; CHECK-LIBCALL-NEXT: cvtsi2ssq %rdi, %xmm0
; CHECK-LIBCALL-NEXT: callq __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: movw %ax, ([[ADDR]])
; CHECK_LIBCALL-NEXT: popq [[ADDR]]
; CHECK_LIBCALL-NEXT: retq

; CHECK-F16C-NEXT: vcvtsi2ssq %rdi, [[REG0:%[a-z0-9]+]], [[REG0]]
; CHECK-F16C-NEXT: vcvtps2ph $0, [[REG0]], [[REG0]]
; CHECK-F16C-NEXT: vmovd [[REG0]], %eax
; CHECK-F16C-NEXT: movw %ax, (%rsi)
; CHECK-F16C-NEXT: retq
  %r = sitofp i64 %a to half
  store half %r, half* %p
  ret void
}

define i64 @test_fptoui_i64(half* %p) #0 {
; CHECK-LABEL: test_fptoui_i64:

; FP_TO_UINT is expanded using FP_TO_SINT
; CHECK-LIBCALL-NEXT: pushq %rax
; CHECK-LIBCALL-NEXT: movzwl (%rdi), %edi
; CHECK-LIBCALL-NEXT: callq __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: movss {{.[A-Z_0-9]+}}(%rip), [[REG1:%[a-z0-9]+]]
; CHECK-LIBCALL-NEXT: movaps %xmm0, [[REG2:%[a-z0-9]+]]
; CHECK-LIBCALL-NEXT: subss [[REG1]], [[REG2]]
; CHECK-LIBCALL-NEXT: cvttss2si [[REG2]], [[REG3:%[a-z0-9]+]]
; CHECK-LIBCALL-NEXT: movabsq  $-9223372036854775808, [[REG4:%[a-z0-9]+]]
; CHECK-LIBCALL-NEXT: xorq [[REG3]], [[REG4]]
; CHECK-LIBCALL-NEXT: cvttss2si %xmm0, [[REG5:%[a-z0-9]+]]
; CHECK-LIBCALL-NEXT: ucomiss [[REG1]], %xmm0
; CHECK-LIBCALL-NEXT: cmovaeq [[REG4]], [[REG5]]
; CHECK-LIBCALL-NEXT: popq %rdx
; CHECK-LIBCALL-NEXT: retq

; CHECK-F16C-NEXT: movswl (%rdi), [[REG0:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vmovd [[REG0]], [[REG1:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vcvtph2ps [[REG1]], [[REG2:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vmovss {{.[A-Z_0-9]+}}(%rip), [[REG3:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vsubss [[REG3]], [[REG2]], [[REG4:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vcvttss2si [[REG4]], [[REG5:%[a-z0-9]+]]
; CHECK-F16C-NEXT: movabsq $-9223372036854775808, [[REG6:%[a-z0-9]+]]
; CHECK-F16C-NEXT: xorq [[REG5]], [[REG6:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vcvttss2si [[REG2]], [[REG7:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vucomiss [[REG3]], [[REG2]]
; CHECK-F16C-NEXT: cmovaeq [[REG6]], %rax
; CHECK-F16C-NEXT: retq
  %a = load half, half* %p, align 2
  %r = fptoui half %a to i64
  ret i64 %r
}

define void @test_uitofp_i64(i64 %a, half* %p) #0 {
; CHECK-LABEL: test_uitofp_i64:
; CHECK-LIBCALL-NEXT: pushq [[ADDR:%[a-z0-9]+]]
; CHECK-LIBCALL-NEXT: movq %rsi, [[ADDR]]
; CHECK-NEXT: movl %edi, [[REG0:%[a-z0-9]+]]
; CHECK-NEXT: andl $1, [[REG0]]
; CHECK-NEXT: testq %rdi, %rdi
; CHECK-NEXT: js [[LABEL1:.LBB[0-9_]+]]

; simple conversion to float if non-negative
; CHECK-LIBCALL-NEXT: cvtsi2ssq %rdi, [[REG1:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vcvtsi2ssq %rdi, [[REG1:%[a-z0-9]+]], [[REG1]]
; CHECK-NEXT: jmp [[LABEL2:.LBB[0-9_]+]]

; convert using shift+or if negative
; CHECK-NEXT: [[LABEL1]]:
; CHECK-NEXT: shrq %rdi
; CHECK-NEXT: orq %rdi, [[REG2:%[a-z0-9]+]]
; CHECK-LIBCALL-NEXT: cvtsi2ssq [[REG2]], [[REG3:%[a-z0-9]+]]
; CHECK-LIBCALL-NEXT: addss [[REG3]], [[REG1]]
; CHECK-F16C-NEXT: vcvtsi2ssq [[REG2]], [[REG3:%[a-z0-9]+]], [[REG3]]
; CHECK-F16C-NEXT: vaddss [[REG3]], [[REG3]], [[REG1:[%a-z0-9]+]]

; convert float to half
; CHECK-NEXT: [[LABEL2]]:
; CHECK-LIBCALL-NEXT: callq __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: movw %ax, ([[ADDR]])
; CHECK-LIBCALL-NEXT: popq [[ADDR]]
; CHECK-F16C-NEXT: vcvtps2ph $0, [[REG1]], [[REG4:%[a-z0-9]+]]
; CHECK-F16C-NEXT: vmovd [[REG4]], %eax
; CHECK-F16C-NEXT: movw %ax, (%rsi)
; CHECK-NEXT: retq

  %r = uitofp i64 %a to half
  store half %r, half* %p
  ret void
}

define <4 x float> @test_extend32_vec4(<4 x half>* %p) #0 {
; CHECK-LABEL: test_extend32_vec4:

; CHECK-LIBCALL: callq __gnu_h2f_ieee
; CHECK-LIBCALL: callq __gnu_h2f_ieee
; CHECK-LIBCALL: callq __gnu_h2f_ieee
; CHECK-LIBCALL: callq __gnu_h2f_ieee
; CHECK-F16C: vcvtph2ps
; CHECK-F16C: vcvtph2ps
; CHECK-F16C: vcvtph2ps
; CHECK-F16C: vcvtph2ps
  %a = load <4 x half>, <4 x half>* %p, align 8
  %b = fpext <4 x half> %a to <4 x float>
  ret <4 x float> %b
}

define <4 x double> @test_extend64_vec4(<4 x half>* %p) #0 {
; CHECK-LABEL: test_extend64_vec4

; CHECK-LIBCALL: callq __gnu_h2f_ieee
; CHECK-LIBCALL-DAG: callq __gnu_h2f_ieee
; CHECK-LIBCALL-DAG: callq __gnu_h2f_ieee
; CHECK-LIBCALL-DAG: callq __gnu_h2f_ieee
; CHECK-LIBCALL-DAG: cvtss2sd
; CHECK-LIBCALL-DAG: cvtss2sd
; CHECK-LIBCALL-DAG: cvtss2sd
; CHECK-LIBCALL: cvtss2sd
; CHECK-F16C: vcvtph2ps
; CHECK-F16C-DAG: vcvtph2ps
; CHECK-F16C-DAG: vcvtph2ps
; CHECK-F16C-DAG: vcvtph2ps
; CHECK-F16C-DAG: vcvtss2sd
; CHECK-F16C-DAG: vcvtss2sd
; CHECK-F16C-DAG: vcvtss2sd
; CHECK-F16C: vcvtss2sd
  %a = load <4 x half>, <4 x half>* %p, align 8
  %b = fpext <4 x half> %a to <4 x double>
  ret <4 x double> %b
}

define void @test_trunc32_vec4(<4 x float> %a, <4 x half>* %p) {
; CHECK-LABEL: test_trunc32_vec4:

; CHECK-LIBCALL: callq __gnu_f2h_ieee
; CHECK-LIBCALL: callq __gnu_f2h_ieee
; CHECK-LIBCALL: callq __gnu_f2h_ieee
; CHECK-LIBCALL: callq __gnu_f2h_ieee
; CHECK-F16C: vcvtps2ph
; CHECK-F16C: vcvtps2ph
; CHECK-F16C: vcvtps2ph
; CHECK-F16C: vcvtps2ph
; CHECK: movw
; CHECK: movw
; CHECK: movw
; CHECK: movw
  %v = fptrunc <4 x float> %a to <4 x half>
  store <4 x half> %v, <4 x half>* %p
  ret void
}

define void @test_trunc64_vec4(<4 x double> %a, <4 x half>* %p) {
; CHECK-LABEL: test_trunc64_vec4:
; CHECK: callq  __truncdfhf2
; CHECK: callq  __truncdfhf2
; CHECK: callq  __truncdfhf2
; CHECK: callq  __truncdfhf2
; CHECK: movw
; CHECK: movw
; CHECK: movw
; CHECK: movw
  %v = fptrunc <4 x double> %a to <4 x half>
  store <4 x half> %v, <4 x half>* %p
  ret void
}

attributes #0 = { nounwind }
