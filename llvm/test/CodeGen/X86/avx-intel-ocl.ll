; RUN: llc < %s -mtriple=i386-pc-win32 -mcpu=corei7-avx -mattr=+avx | FileCheck -check-prefix=WIN32 %s
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=corei7-avx -mattr=+avx | FileCheck -check-prefix=WIN64 %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck -check-prefix=NOT_WIN %s

declare <16 x float> @func_float16_ptr(<16 x float>, <16 x float> *)
declare <16 x float> @func_float16(<16 x float>, <16 x float>)
; WIN64: testf16_inp
; WIN64: vaddps  {{.*}}, {{%ymm[0-1]}}
; WIN64: vaddps  {{.*}}, {{%ymm[0-1]}}
; WIN64: leaq    {{.*}}(%rsp), %rcx
; WIN64: call
; WIN64: ret

; WIN32: testf16_inp
; WIN32: movl    %eax, (%esp)
; WIN32: vaddps  {{.*}}, {{%ymm[0-1]}}
; WIN32: vaddps  {{.*}}, {{%ymm[0-1]}}
; WIN32: call
; WIN32: ret

; NOT_WIN: testf16_inp
; NOT_WIN: vaddps  {{.*}}, {{%ymm[0-1]}}
; NOT_WIN: vaddps  {{.*}}, {{%ymm[0-1]}}
; NOT_WIN: leaq    {{.*}}(%rsp), %rdi
; NOT_WIN: call
; NOT_WIN: ret

;test calling conventions - input parameters
define <16 x float> @testf16_inp(<16 x float> %a, <16 x float> %b) nounwind {
  %y = alloca <16 x float>, align 16
  %x = fadd <16 x float> %a, %b
  %1 = call intel_ocl_bicc <16 x float> @func_float16_ptr(<16 x float> %x, <16 x float>* %y) 
  %2 = load <16 x float>* %y, align 16
  %3 = fadd <16 x float> %2, %1
  ret <16 x float> %3
}

;test calling conventions - preserved registers

; preserved ymm6-ymm15
; WIN64: testf16_regs
; WIN64: call
; WIN64: vaddps  {{%ymm[6-7]}}, %ymm0, %ymm0
; WIN64: vaddps  {{%ymm[6-7]}}, %ymm1, %ymm1
; WIN64: ret

; preserved ymm8-ymm15
; NOT_WIN: testf16_regs
; NOT_WIN: call
; NOT_WIN: vaddps  {{%ymm[8-9]}}, %ymm0, %ymm0
; NOT_WIN: vaddps  {{%ymm[8-9]}}, %ymm1, %ymm1
; NOT_WIN: ret

define <16 x float> @testf16_regs(<16 x float> %a, <16 x float> %b) nounwind {
  %y = alloca <16 x float>, align 16
  %x = fadd <16 x float> %a, %b
  %1 = call intel_ocl_bicc <16 x float> @func_float16_ptr(<16 x float> %x, <16 x float>* %y) 
  %2 = load <16 x float>* %y, align 16
  %3 = fadd <16 x float> %1, %b
  %4 = fadd <16 x float> %2, %3
  ret <16 x float> %4
}

; test calling conventions - prolog and epilog
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: vmovaps {{%ymm([6-9]|1[0-5])}}, {{.*(%rsp).*}}     # 32-byte Spill
; WIN64: call
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload
; WIN64: vmovaps {{.*(%rsp).*}}, {{%ymm([6-9]|1[0-5])}}     # 32-byte Reload

; NOT_WIN: vmovaps {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rbp)  ## 32-byte Spill
; NOT_WIN: vmovaps {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rbp)  ## 32-byte Spill
; NOT_WIN: vmovaps {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rbp)  ## 32-byte Spill
; NOT_WIN: vmovaps {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rbp)  ## 32-byte Spill
; NOT_WIN: vmovaps {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rbp)  ## 32-byte Spill
; NOT_WIN: vmovaps {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rbp)  ## 32-byte Spill
; NOT_WIN: vmovaps {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rbp)  ## 32-byte Spill
; NOT_WIN: vmovaps {{%ymm([8-9]|1[0-5])}}, {{.*}}(%rbp)  ## 32-byte Spill
; NOT_WIN: call
; NOT_WIN: vmovaps {{.*}}(%rbp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; NOT_WIN: vmovaps {{.*}}(%rbp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; NOT_WIN: vmovaps {{.*}}(%rbp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; NOT_WIN: vmovaps {{.*}}(%rbp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; NOT_WIN: vmovaps {{.*}}(%rbp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; NOT_WIN: vmovaps {{.*}}(%rbp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; NOT_WIN: vmovaps {{.*}}(%rbp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
; NOT_WIN: vmovaps {{.*}}(%rbp), {{%ymm([8-9]|1[0-5])}} ## 32-byte Reload
define intel_ocl_bicc <16 x float> @test_prolog_epilog(<16 x float> %a, <16 x float> %b) nounwind {
   %c = call <16 x float> @func_float16(<16 x float> %a, <16 x float> %b)
   ret <16 x float> %c
}
