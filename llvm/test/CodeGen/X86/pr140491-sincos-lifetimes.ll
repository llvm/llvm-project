; RUN: llc < %s | FileCheck %s

; This test is reduced from https://github.com/llvm/llvm-project/issues/140491.
; It checks that when `@llvm.sincos.f32` is expanded to a call to
; `sincosf(float, float* out_sin, float* out_cos)` and the store of `%cos` to
; `%computed` is folded into the `sincosf` call. The use of `%cos`in the later
; `fneg %cos` -- which expands to a load of `%computed`, will perform the load
; before the `@llvm.lifetime.end.p0(%computed)` to ensure the correct value is
; taken for `%cos`.

target triple = "x86_64-sie-ps5"

declare void @use_ptr(ptr readonly)

define i32 @sincos_stack_slot_with_lifetime(float %in)  {
; CHECK-LABEL: sincos_stack_slot_with_lifetime:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pushq %rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    subq $32, %rsp
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    .cfi_offset %rbx, -16
; CHECK-NEXT:    leaq 12(%rsp), %rdi
; CHECK-NEXT:    leaq 8(%rsp), %rbx
; CHECK-NEXT:    movq %rbx, %rsi
; CHECK-NEXT:    callq sincosf@PLT
; CHECK-NEXT:    movss 8(%rsp), %xmm0 # xmm0 = mem[0],zero,zero,zero
; CHECK-NEXT:    movaps %xmm0, 16(%rsp) # 16-byte Spill
; CHECK-NEXT:    movq %rbx, %rdi
; CHECK-NEXT:    callq use_ptr
; CHECK-NEXT:    movss 12(%rsp), %xmm0 # xmm0 = mem[0],zero,zero,zero
; CHECK-NEXT:    xorps {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; CHECK-NEXT:    movss %xmm0, 8(%rsp)
; CHECK-NEXT:    leaq 8(%rsp), %rdi
; CHECK-NEXT:    callq use_ptr
; CHECK-NEXT:    movaps 16(%rsp), %xmm0 # 16-byte Reload
; CHECK-NEXT:    xorps {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; CHECK-NEXT:    movss %xmm0, 8(%rsp)
; CHECK-NEXT:    leaq 8(%rsp), %rdi
; CHECK-NEXT:    callq use_ptr
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    addq $32, %rsp
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    popq %rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    retq
entry:
  %computed = alloca float, align 4
  %computed1 = alloca float, align 4
  %computed3 = alloca float, align 4
  %sincos = tail call { float, float } @llvm.sincos.f32(float %in)
  %sin = extractvalue { float, float } %sincos, 0
  %cos = extractvalue { float, float } %sincos, 1
  call void @llvm.lifetime.start.p0(ptr nonnull %computed)
  store float %cos, ptr %computed, align 4
  call void @use_ptr(ptr nonnull %computed)
  call void @llvm.lifetime.end.p0(ptr nonnull %computed)
  call void @llvm.lifetime.start.p0(ptr nonnull %computed1)
  %fneg_sin = fneg float %sin
  store float %fneg_sin, ptr %computed1, align 4
  call void @use_ptr(ptr nonnull %computed1)
  call void @llvm.lifetime.end.p0(ptr nonnull %computed1)
  call void @llvm.lifetime.start.p0(ptr nonnull %computed3)
  %fneg_cos = fneg float %cos
  store float %fneg_cos, ptr %computed3, align 4
  call void @use_ptr(ptr nonnull %computed3)
  call void @llvm.lifetime.end.p0(ptr nonnull %computed3)
  ret i32 0
}

