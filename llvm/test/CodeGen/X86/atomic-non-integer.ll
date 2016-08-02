; RUN: llc < %s -mtriple=x86_64-linux-generic -verify-machineinstrs -mattr=sse2 | FileCheck %s

; Note: This test is testing that the lowering for atomics matches what we
; currently emit for non-atomics + the atomic restriction.  The presence of
; particular lowering detail in these tests should not be read as requiring
; that detail for correctness unless it's related to the atomicity itself.
; (Specifically, there were reviewer questions about the lowering for halfs
;  and their calling convention which remain unresolved.)

define void @store_half(half* %fptr, half %v) {
; CHECK-LABEL: @store_half
; CHECK: movq	%rdi, %rbx
; CHECK: callq	__gnu_f2h_ieee
; CHECK: movw	%ax, (%rbx)
  store atomic half %v, half* %fptr unordered, align 2
  ret void
}

define void @store_float(float* %fptr, float %v) {
; CHECK-LABEL: @store_float
; CHECK: movd	%xmm0, %eax
; CHECK: movl	%eax, (%rdi)
  store atomic float %v, float* %fptr unordered, align 4
  ret void
}

define void @store_double(double* %fptr, double %v) {
; CHECK-LABEL: @store_double
; CHECK: movd	%xmm0, %rax
; CHECK: movq	%rax, (%rdi)
  store atomic double %v, double* %fptr unordered, align 8
  ret void
}

define void @store_fp128(fp128* %fptr, fp128 %v) {
; CHECK-LABEL: @store_fp128
; CHECK: callq	__sync_lock_test_and_set_16
  store atomic fp128 %v, fp128* %fptr unordered, align 16
  ret void
}

define half @load_half(half* %fptr) {
; CHECK-LABEL: @load_half
; CHECK: movw	(%rdi), %ax
; CHECK: movzwl	%ax, %edi
; CHECK: jmp	__gnu_h2f_ieee
  %v = load atomic half, half* %fptr unordered, align 2
  ret half %v
}

define float @load_float(float* %fptr) {
; CHECK-LABEL: @load_float
; CHECK: movl	(%rdi), %eax
; CHECK: movd	%eax, %xmm0
  %v = load atomic float, float* %fptr unordered, align 4
  ret float %v
}

define double @load_double(double* %fptr) {
; CHECK-LABEL: @load_double
; CHECK: movq	(%rdi), %rax
; CHECK: movd	%rax, %xmm0
  %v = load atomic double, double* %fptr unordered, align 8
  ret double %v
}

define fp128 @load_fp128(fp128* %fptr) {
; CHECK-LABEL: @load_fp128
; CHECK: callq	__sync_val_compare_and_swap_16
  %v = load atomic fp128, fp128* %fptr unordered, align 16
  ret fp128 %v
}


; sanity check the seq_cst lowering since that's the 
; interesting one from an ordering perspective on x86.

define void @store_float_seq_cst(float* %fptr, float %v) {
; CHECK-LABEL: @store_float_seq_cst
; CHECK: movd	%xmm0, %eax
; CHECK: xchgl	%eax, (%rdi)
  store atomic float %v, float* %fptr seq_cst, align 4
  ret void
}

define void @store_double_seq_cst(double* %fptr, double %v) {
; CHECK-LABEL: @store_double_seq_cst
; CHECK: movd	%xmm0, %rax
; CHECK: xchgq	%rax, (%rdi)
  store atomic double %v, double* %fptr seq_cst, align 8
  ret void
}

define float @load_float_seq_cst(float* %fptr) {
; CHECK-LABEL: @load_float_seq_cst
; CHECK: movl	(%rdi), %eax
; CHECK: movd	%eax, %xmm0
  %v = load atomic float, float* %fptr seq_cst, align 4
  ret float %v
}

define double @load_double_seq_cst(double* %fptr) {
; CHECK-LABEL: @load_double_seq_cst
; CHECK: movq	(%rdi), %rax
; CHECK: movd	%rax, %xmm0
  %v = load atomic double, double* %fptr seq_cst, align 8
  ret double %v
}
