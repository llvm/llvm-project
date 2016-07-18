; RUN: llc < %s -mtriple=i686-apple-darwin -mcpu=knl | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -mtriple=i386-pc-win32 -mcpu=knl | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=knl | FileCheck -check-prefix=WIN64 %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck -check-prefix=X64 %s

declare <16 x float> @func_float16_ptr(<16 x float>, <16 x float> *)
declare <16 x float> @func_float16(<16 x float>, <16 x float>)
declare i32 @func_int(i32, i32)

; WIN64-LABEL: testf16_inp
; WIN64: vaddps  {{.*}}, {{%zmm[0-1]}}
; WIN64: leaq    {{.*}}(%rsp), %rcx
; WIN64: call
; WIN64: ret

; X32-LABEL: testf16_inp
; X32: vaddps  {{.*}}, {{%zmm[0-1]}}
; Push is not deemed profitable if we're realigning the stack.
; X32: {{pushl|movl}}   %eax
; X32: call
; X32: ret

; X64-LABEL: testf16_inp
; X64: vaddps  {{.*}}, {{%zmm[0-1]}}
; X64: leaq    {{.*}}(%rsp), %rdi
; X64: call
; X64: ret

;test calling conventions - input parameters
define <16 x float> @testf16_inp(<16 x float> %a, <16 x float> %b) nounwind {
  %y = alloca <16 x float>, align 16
  %x = fadd <16 x float> %a, %b
  %1 = call intel_ocl_bicc <16 x float> @func_float16_ptr(<16 x float> %x, <16 x float>* %y)
  %2 = load <16 x float>, <16 x float>* %y, align 16
  %3 = fadd <16 x float> %2, %1
  ret <16 x float> %3
}

;test calling conventions - preserved registers

; preserved zmm16-
; WIN64-LABEL: testf16_regs
; WIN64: call
; WIN64: vaddps  %zmm16, %zmm0, %zmm0
; WIN64: ret

; preserved zmm16-
; X64-LABEL: testf16_regs
; X64: call
; X64: vaddps  %zmm16, %zmm0, %zmm0
; X64: ret

define <16 x float> @testf16_regs(<16 x float> %a, <16 x float> %b) nounwind {
  %y = alloca <16 x float>, align 16
  %x = fadd <16 x float> %a, %b
  %1 = call intel_ocl_bicc <16 x float> @func_float16_ptr(<16 x float> %x, <16 x float>* %y)
  %2 = load <16 x float>, <16 x float>* %y, align 16
  %3 = fadd <16 x float> %1, %b
  %4 = fadd <16 x float> %2, %3
  ret <16 x float> %4
}

; test calling conventions - prolog and epilog
; WIN64-LABEL: test_prolog_epilog
; WIN64: vmovups %zmm21, {{.*(%rbp).*}}     # 64-byte Spill
; WIN64: vmovups %zmm6, {{.*(%rbp).*}}     # 64-byte Spill
; WIN64: call
; WIN64: vmovups {{.*(%rbp).*}}, %zmm6      # 64-byte Reload
; WIN64: vmovups {{.*(%rbp).*}}, %zmm21     # 64-byte Reload

; X64-LABEL: test_prolog_epilog
; X64:  kmovq   %k7, {{.*}}(%rsp)         ## 8-byte Spill
; X64:  kmovq   %k6, {{.*}}(%rsp)         ## 8-byte Spill
; X64:  kmovq   %k5, {{.*}}(%rsp)         ## 8-byte Spill
; X64:  kmovq   %k4, {{.*}}(%rsp)         ## 8-byte Spill
; X64: vmovups %zmm31, {{.*}}(%rsp)  ## 64-byte Spill
; X64: vmovups %zmm16, {{.*}}(%rsp)  ## 64-byte Spill
; X64: call
; X64: vmovups {{.*}}(%rsp), %zmm16 ## 64-byte Reload
; X64: vmovups {{.*}}(%rsp), %zmm31 ## 64-byte Reload
define intel_ocl_bicc <16 x float> @test_prolog_epilog(<16 x float> %a, <16 x float> %b) nounwind {
   %c = call <16 x float> @func_float16(<16 x float> %a, <16 x float> %b)
   ret <16 x float> %c
}


declare <16 x float> @func_float16_mask(<16 x float>, <16 x i1>)

; X64-LABEL: testf16_inp_mask
; X64: kmovw   %edi, %k1
; X64: call
define <16 x float> @testf16_inp_mask(<16 x float> %a, i16 %mask)  {
  %imask = bitcast i16 %mask to <16 x i1>
  %1 = call intel_ocl_bicc <16 x float> @func_float16_mask(<16 x float> %a, <16 x i1> %imask)
  ret <16 x float> %1
}

; X64-LABEL: test_prolog_epilog_with_mask
; X64: kxorw   %k{{.*}}, %k{{.*}}, %k1
; X64: call
define intel_ocl_bicc <16 x float> @test_prolog_epilog_with_mask(<16 x float> %a, <16 x i32> %x1, <16 x i32>%x2, <16 x i1> %mask) nounwind {
   %cmp_res = icmp eq <16 x i32>%x1, %x2
   %mask1 = xor <16 x i1> %cmp_res, %mask
   %c = call intel_ocl_bicc <16 x float> @func_float16_mask(<16 x float> %a, <16 x i1>%mask1)
   ret <16 x float> %c
}
