; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=LINUX
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=WIN64

; In a void preserve_mostcc function, %rax is a callee-saved register (not an
; entry live-in). Verify that isEAXLiveIn treats %rax as live so >2GB stack
; frame adjustments in emitSPUpdate do not clobber %rax in the prologue.
define preserve_mostcc void @large_stack_preserve_most() nounwind {
; LINUX-LABEL: large_stack_preserve_most:
; LINUX-NOT:   movabsq {{.*}}, %rax
; LINUX:       subq $2147483647, %rsp
; LINUX-NEXT:  subq $852516249, %rsp
; LINUX-NEXT:  movb $1, -112(%rsp)
; LINUX-NEXT:  movl $2999999896, %r11d
; LINUX-NEXT:  addq %r11, %rsp
; LINUX-NEXT:  retq
  %a = alloca [3000000000 x i8], align 16
  store volatile i8 1, ptr %a, align 16
  ret void
}

; Verify that isEAXLiveIn also causes emitStackProbeCall to save and restore
; callee-saved %rax around __probestack in a void preserve_mostcc function.
define preserve_mostcc void @probe_stack_call_preserve_most() "probe-stack"="__probestack" nounwind {
; LINUX-LABEL: probe_stack_call_preserve_most:
; LINUX:       pushq %rax
; LINUX-NEXT:  movl $8192, %eax
; LINUX-NEXT:  callq __probestack
; LINUX-NEXT:  subq %rax, %rsp
; LINUX-NEXT:  movq 8192(%rsp), %rax
; LINUX:       retq
  %a = alloca [8192 x i8], align 16
  store volatile i8 1, ptr %a, align 16
  ret void
}

; In preserve_nonecc, %r11 holds the 11th GPR argument. Verify that
; emitStackProbeInlineGenericLoop selects an available scratch register (%r10
; or %rax) or saves/restores %r11 when all of %r11, %r10, and %rax are in use.
define preserve_nonecc i64 @inline_probe_preserve_none_11args(i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11) "probe-stack"="inline-asm" nounwind {
; LINUX-LABEL: inline_probe_preserve_none_11args:
; LINUX:       movq %rsp, %r10
; LINUX-NEXT:  subq $36864, %r10
; LINUX:       cmpq %r10, %rsp
; LINUX:       movq %r11, %rax
; LINUX:       retq
  %a = alloca [40960 x i8], align 16
  store volatile i8 1, ptr %a, align 16
  ret i64 %a11
}

define preserve_nonecc i64 @inline_probe_preserve_none_11args_nest(i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %unused_a12, ptr nest %chain) "probe-stack"="inline-asm" nounwind {
; LINUX-LABEL: inline_probe_preserve_none_11args_nest:
; LINUX:       movq %rsp, %rax
; LINUX-NEXT:  subq $36864, %rax
; LINUX:       cmpq %rax, %rsp
; LINUX:       leaq (%r11,%r10), %rax
; LINUX:       retq
  %a = alloca [40960 x i8], align 16
  store volatile i8 1, ptr %a, align 16
  %c = ptrtoint ptr %chain to i64
  %sum = add i64 %a11, %c
  ret i64 %sum
}

define preserve_nonecc i64 @inline_probe_preserve_none_12args_nest(i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, ptr nest %chain) "probe-stack"="inline-asm" nounwind {
; LINUX-LABEL: inline_probe_preserve_none_12args_nest:
; LINUX:       subq $4096, %rsp
; LINUX-NEXT:  movq %r11, (%rsp)
; LINUX-NEXT:  movq %rsp, %r11
; LINUX-NEXT:  subq $32768, %r11
; LINUX:       cmpq %r11, %rsp
; LINUX:       subq $3976, %rsp
; LINUX-NEXT:  movq 36744(%rsp), %r11
; LINUX:       retq
  %a = alloca [40960 x i8], align 16
  store volatile i8 1, ptr %a, align 16
  %c = ptrtoint ptr %chain to i64
  %s1 = add i64 %a11, %a12
  %s2 = add i64 %s1, %c
  ret i64 %s2
}

; Verify that has128ByteRedZone disables the 128-byte red zone on Windows
; x86_64 targets regardless of calling convention (including preserve_mostcc
; and preserve_nonecc).
define preserve_mostcc void @win64_no_redzone_preserve_most() nounwind {
; WIN64-LABEL: win64_no_redzone_preserve_most:
; WIN64:       pushq %rax
; WIN64-NEXT:  movl $42, 4(%rsp)
; WIN64-NEXT:  popq %r11
; WIN64-NEXT:  retq
  %a = alloca i32, align 4
  store volatile i32 42, ptr %a, align 4
  ret void
}

define preserve_nonecc void @win64_no_redzone_preserve_none() nounwind {
; WIN64-LABEL: win64_no_redzone_preserve_none:
; WIN64:       pushq %rax
; WIN64-NEXT:  movl $42, 4(%rsp)
; WIN64-NEXT:  popq %rax
; WIN64-NEXT:  retq
  %a = alloca i32, align 4
  store volatile i32 42, ptr %a, align 4
  ret void
}
