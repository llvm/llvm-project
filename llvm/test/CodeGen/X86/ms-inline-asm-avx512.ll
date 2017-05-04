; RUN: llc < %s | FileCheck %s

; Generated from clang/test/CodeGen/ms-inline-asm-avx512.c

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Function Attrs: noinline nounwind
define void @ignore_fe_size() #0 {
entry:
  %c = alloca i8, align 1
  call void asm sideeffect inteldialect "vaddps xmm1, xmm2, $1{1to4}\0A\09vaddps xmm1, xmm2, $2\0A\09mov eax, $3\0A\09mov $0, rax", "=*m,*m,*m,*m,~{eax},~{xmm1},~{dirflag},~{fpsr},~{flags}"(i8* %c, i8* %c, i8* %c, i8* %c) #1
  ret void
}

; CHECK-LABEL: ignore_fe_size:
; CHECK: vaddps  7(%rsp){1to4}, %xmm2, %xmm1
; CHECK: vaddps  7(%rsp), %xmm2, %xmm1
; CHECK: movl    7(%rsp), %eax
; CHECK: movq    %rax, 7(%rsp)
; CHECK: retq

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="skylake-avx512" "target-features"="+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+mpx,+pclmul,+pku,+popcnt,+rdrnd,+rdseed,+rtm,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
