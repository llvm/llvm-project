; RUN: llc < %s -mtriple=i686-apple-darwin -mattr=+avx | FileCheck -check-prefix=X86 %s
; RUN: llc < %s -mtriple=i386-pc-win32 -mattr=+avx | FileCheck -check-prefix=X86 %s
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+avx | FileCheck -check-prefix=WIN64 %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+avx | FileCheck -check-prefix=X64 %s

declare <16 x float> @func_float16_ptr(<16 x float>, ptr)
declare <16 x float> @func_float16(<16 x float>, <16 x float>)
declare i32 @func_int(i32, i32)

; WIN64-LABEL: testf16_inp
; WIN64: vaddps  {{.*}}, {{%ymm[0-1]}}
; WIN64: vaddps  {{.*}}, {{%ymm[0-1]}}
; WIN64: leaq    {{.*}}(%rsp), %rcx
; WIN64: call
; WIN64: ret

; X86-LABEL: testf16_inp
; X86: vaddps  {{.*}}, {{%ymm[0-1]}}
; X86: vaddps  {{.*}}, {{%ymm[0-1]}}
; Push is not deemed profitable if we're realigning the stack.
; X86: {{pushl|movl}}   %eax
; X86: call
; X86: ret

; X64-LABEL: testf16_inp
; X64: vaddps  {{.*}}, {{%ymm[0-1]}}
; X64: vaddps  {{.*}}, {{%ymm[0-1]}}
; X64: movq    %rsp, %rdi
; X64: call
; X64: ret

;test calling conventions - input parameters
define <16 x float> @testf16_inp(<16 x float> %a, <16 x float> %b) nounwind {
  %y = alloca <16 x float>, align 16
  %x = fadd <16 x float> %a, %b
  %1 = call intel_ocl_bicc <16 x float> @func_float16_ptr(<16 x float> %x, ptr %y)
  %2 = load <16 x float>, ptr %y, align 16
  %3 = fadd <16 x float> %2, %1
  ret <16 x float> %3
}

;test calling conventions - preserved registers

; preserved ymm6-ymm15
; WIN64-LABEL: testf16_regs
; WIN64: call
; WIN64: vaddps  {{%ymm[6-7]}}, {{%ymm[0-1]}}, {{%ymm[0-1]}}
; WIN64: vaddps  {{%ymm[6-7]}}, {{%ymm[0-1]}}, {{%ymm[0-1]}}
; WIN64: ret

; preserved ymm8-ymm15
; X64-LABEL: testf16_regs
; X64: call
; X64: vaddps  {{%ymm[0-1]}}, {{%ymm[8-9]}}, {{%ymm[0-1]}}
; X64: vaddps  {{%ymm[0-1]}}, {{%ymm[8-9]}}, {{%ymm[0-1]}}
; X64: ret

define <16 x float> @testf16_regs(<16 x float> %a, <16 x float> %b) nounwind {
  %y = alloca <16 x float>, align 16
  %x = fadd <16 x float> %a, %b
  %1 = call intel_ocl_bicc <16 x float> @func_float16_ptr(<16 x float> %x, ptr %y)
  %2 = load <16 x float>, ptr %y, align 16
  %3 = fadd <16 x float> %1, %b
  %4 = fadd <16 x float> %2, %3
  ret <16 x float> %4
}

; test calling conventions - prolog and epilog
; WIN64-LABEL: test_prolog_epilog
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: vmovups {{%ymm([6-9]|1[0-5])}}, {{.*(%rbp).*}}     # 32-byte Spill
; WIN64: call
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload

; X64-LABEL: test_prolog_epilog
; X64: vmovups {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rsp)  ## 32-byte Spill
; X64: vmovups {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rsp)  ## 32-byte Spill
; X64: vmovups {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rsp)  ## 32-byte Spill
; X64: vmovups {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rsp)  ## 32-byte Spill
; X64: vmovups {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rsp)  ## 32-byte Spill
; X64: vmovups {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rsp)  ## 32-byte Spill
; X64: vmovups {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rsp)  ## 32-byte Spill
; X64: vmovups {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rsp)  ## 32-byte Spill
; X64: call
; X64: vmovups {{.*}}(%rsp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; X64: vmovups {{.*}}(%rsp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; X64: vmovups {{.*}}(%rsp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; X64: vmovups {{.*}}(%rsp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; X64: vmovups {{.*}}(%rsp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; X64: vmovups {{.*}}(%rsp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; X64: vmovups {{.*}}(%rsp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; X64: vmovups {{.*}}(%rsp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
define intel_ocl_bicc <16 x float> @test_prolog_epilog(<16 x float> %a, <16 x float> %b) nounwind {
   %c = call <16 x float> @func_float16(<16 x float> %a, <16 x float> %b)
   ret <16 x float> %c
}

; test functions with integer parameters
; pass parameters on stack for 32-bit platform
; X86-LABEL: test_int
; X86: pushl {{.*}}
; X86: pushl {{.*}}
; X86: call
; X86: addl {{.*}}, %eax

; pass parameters in registers for 64-bit platform
; X64-LABEL: test_int
; X64: movl {{.*}}, %esi
; X64: leal {{.*}}, %edi
; X64: call
; X64: addl {{.*}}, %eax
define i32 @test_int(i32 %a, i32 %b) nounwind {
    %c1 = add i32 %a, %b
	%c2 = call intel_ocl_bicc i32 @func_int(i32 %c1, i32 %a)
    %c = add i32 %c2, %b
	ret i32 %c
}

; WIN64-LABEL: test_float4
; WIN64-NOT: vzeroupper
; WIN64: call
; WIN64-NOT: vzeroupper
; WIN64: call
; WIN64: ret

; X64-LABEL: test_float4
; X64-NOT: vzeroupper
; X64: call
; X64-NOT: vzeroupper
; X64: call
; X64: ret

; X86-LABEL: test_float4
; X86: vzeroupper
; X86: call
; X86: vzeroupper
; X86: call
; X86: ret

declare <4 x float> @func_float4(<4 x float>, <4 x float>, <4 x float>)

define <8 x float> @test_float4(<8 x float> %a, <8 x float> %b, <8 x float> %c) nounwind readnone {
entry:
  %0 = shufflevector <8 x float> %a, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1 = shufflevector <8 x float> %b, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %2 = shufflevector <8 x float> %c, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %call.i = tail call intel_ocl_bicc <4 x float> @func_float4(<4 x float> %0, <4 x float> %1, <4 x float> %2) nounwind
  %3 = shufflevector <4 x float> %call.i, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %4 = shufflevector <8 x float> %a, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %5 = shufflevector <8 x float> %b, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %6 = shufflevector <8 x float> %c, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %call.i2 = tail call intel_ocl_bicc <4 x float> @func_float4(<4 x float> %4, <4 x float> %5, <4 x float> %6) nounwind
  %7 = shufflevector <4 x float> %call.i2, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %8 = shufflevector <8 x float> %3, <8 x float> %7, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  ret <8 x float> %8
}
