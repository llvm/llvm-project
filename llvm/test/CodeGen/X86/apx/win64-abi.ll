; RUN: llc -mtriple=x86_64-windows-msvc -mattr=+egpr < %s | FileCheck %s
; RUN: not llc -mtriple=x86_64-windows-msvc -mattr=+egpr,-sse < %s

declare void @external()

; A callee that clobbers all EGPR registers must preserve nonvolatile R30/R31
; and mustn't preserve volatile R16-R29.
define void @test_callee_clobbers_all_egprs() nounwind {
; CHECK-LABEL: test_callee_clobbers_all_egprs:
; CHECK-NOT: pushq %r16
; CHECK-NOT: pushq %r17
; CHECK-NOT: pushq %r18
; CHECK-NOT: pushq %r19
; CHECK-NOT: pushq %r20
; CHECK-NOT: pushq %r21
; CHECK-NOT: pushq %r22
; CHECK-NOT: pushq %r23
; CHECK-NOT: pushq %r24
; CHECK-NOT: pushq %r25
; CHECK-NOT: pushq %r26
; CHECK-NOT: pushq %r27
; CHECK-NOT: pushq %r28
; CHECK-NOT: pushq %r29
; CHECK-DAG: pushq %r30
; CHECK-DAG: pushq %r31
; CHECK: callq external
; CHECK-DAG: popq %r31
; CHECK-DAG: popq %r30
; CHECK-NOT: popq %r29
; CHECK-NOT: popq %r28
; CHECK-NOT: popq %r27
; CHECK-NOT: popq %r26
; CHECK-NOT: popq %r25
; CHECK-NOT: popq %r24
; CHECK-NOT: popq %r23
; CHECK-NOT: popq %r22
; CHECK-NOT: popq %r21
; CHECK-NOT: popq %r20
; CHECK-NOT: popq %r19
; CHECK-NOT: popq %r18
; CHECK-NOT: popq %r17
; CHECK-NOT: popq %r16
; CHECK: retq
  call void @external()
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; A caller should NOT save/restore R30/R31 around calls (they are preserved
; by the callee).
define void @test_caller_no_save() nounwind {
; CHECK-LABEL: test_caller_no_save:
; CHECK-NOT: pushq %r30
; CHECK-NOT: pushq %r31
; CHECK: callq external
; CHECK-NOT: popq %r31
; CHECK-NOT: popq %r30
; CHECK: retq
  call void @external()
  ret void
}

; PreserveMost CC with EGPR must preserve R30 and R31.
define preserve_mostcc void @test_preservemost_cc_apx_abi() nounwind {
; CHECK-LABEL: test_preservemost_cc_apx_abi:
; CHECK-DAG: pushq %r30
; CHECK-DAG: pushq %r31
; CHECK-DAG: popq %r31
; CHECK-DAG: popq %r30
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; Intel OCL BI AVX CC with EGPR must preserve R30 and R31.
define intel_ocl_bicc void @test_intel_ocl_bicc_avx_apx_abi() nounwind "target-features"="+avx" {
; CHECK-LABEL: test_intel_ocl_bicc_avx_apx_abi:
; CHECK-DAG: pushq %r30
; CHECK-DAG: pushq %r31
; CHECK-DAG: popq %r31
; CHECK-DAG: popq %r30
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; Intel OCL BI AVX512 CC with EGPR must preserve R30 and R31.
define intel_ocl_bicc void @test_intel_ocl_bicc_apx_avx512_abi() nounwind "target-features"="+avx512f" {
; CHECK-LABEL: test_intel_ocl_bicc_apx_avx512_abi:
; CHECK-DAG: pushq %r30
; CHECK-DAG: pushq %r31
; CHECK-DAG: popq %r31
; CHECK-DAG: popq %r30
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; RegCall CC with EGPR must preserve R30 and R31.
define x86_regcallcc void @test_x86_regcallcc_apx_abi() nounwind {
; CHECK-LABEL: test_x86_regcallcc_apx_abi:
; CHECK-DAG: pushq %r30
; CHECK-DAG: pushq %r31
; CHECK-DAG: popq %r31
; CHECK-DAG: popq %r30
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; SwiftError CC with EGPR must preserve R30 and R31, but NOT R12.
define void @test_swifterror_apx_abi(ptr swifterror %err) nounwind {
; CHECK-LABEL: test_swifterror_apx_abi:
; CHECK-DAG: pushq %r30
; CHECK-DAG: pushq %r31
; CHECK-NOT: pushq %r12
; CHECK-DAG: popq %r31
; CHECK-DAG: popq %r30
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; SwiftTail CC with EGPR must preserve R30 and R31, but NOT R13 or R14.
define swifttailcc void @test_swifttailcc_apx_abi() nounwind {
; CHECK-LABEL: test_swifttailcc_apx_abi:
; CHECK-DAG: pushq %r30
; CHECK-DAG: pushq %r31
; CHECK-NOT: pushq %r13
; CHECK-NOT: pushq %r14
; CHECK-DAG: popq %r31
; CHECK-DAG: popq %r30
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; CFGuard dispatch must preserve R30 (the check stub never touches EGPR).
; The CFGuard check call should NOT cause an extra spill of R30.

declare i32 @target_func()

define i64 @test_cfguard_preserves_r30() nounwind {
; CHECK-LABEL: test_cfguard_preserves_r30:
; CHECK: pushq %r30
; CHECK: __guard_dispatch_icall_fptr
; CHECK-NOT: movq %r30, {{.*}}(%rsp)
; CHECK: popq %r30
; CHECK: retq
entry:
  %val = call i64 asm sideeffect "", "={r30}"()
  %fptr = alloca ptr, align 8
  store ptr @target_func, ptr %fptr, align 8
  %0 = load ptr, ptr %fptr, align 8
  %1 = call i32 %0()
  ret i64 %val
}

; PUSH2/POP2 must be used for callee-saved registers including R30+R31 pair.
define void @test_push2pop2_r30_r31() nounwind "target-features"="+egpr,+push2pop2" "frame-pointer"="all" {
; CHECK-LABEL: test_push2pop2_r30_r31:
; CHECK: push2 %r30, %r31
; CHECK: callq external
; CHECK: pop2 %r31, %r30
; CHECK: retq
  call void @external()
  call void asm sideeffect "", "~{rax},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
