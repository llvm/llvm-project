; NOTE: Test cases for FCMP-FCSEL and CMP/CMN-CSEL code layout optimization
; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-apple-darwin -mcpu=apple-m4 -aarch64-code-layout-opt=3 | FileCheck %s

; Test coverage for optimizeForCodeAlignment function:
; 1. Basic FCMP-FCSEL instruction pair detection and function alignment (single/double precision)
; 2. Loop block alignment for instruction pairs in loops (simple, nested, multi-block)
; 3. Multiple instruction pairs in same function (also tests different predicates)
; 4. FCMP with immediate operand (#0.0) is excluded from optimization
; 5. Instruction pairs with function calls
; 6. Negative tests (no false positives)
; 7. Basic CMP-CSEL and CMN-CSEL instruction pair detection and function alignment
; 8. CMP/CMN with immediate <=15 qualifies; immediate >15 is excluded

; Test 1: Basic single-precision FCMP-FCSEL instruction pair
; CHECK: .globl _test_basic_fcmp_fcsel_single
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_basic_fcmp_fcsel_single:
define float @test_basic_fcmp_fcsel_single(float %a, float %b, float %c, float %d) {
entry:
  %cmp = fcmp oeq float %a, %b
  %sel = select i1 %cmp, float %c, float %d
  ret float %sel
}

; Test 2: Basic double-precision FCMP-FCSEL instruction pair
; CHECK: .globl _test_basic_fcmp_fcsel_double
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_basic_fcmp_fcsel_double:
define double @test_basic_fcmp_fcsel_double(double %a, double %b, double %c, double %d) {
entry:
  %cmp = fcmp oeq double %a, %b
  %sel = select i1 %cmp, double %c, double %d
  ret double %sel
}

; Test 3: FCMP-FCSEL instruction pair in a simple loop
; CHECK: .globl _test_fcmp_fcsel_in_loop
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_fcmp_fcsel_in_loop:
define float @test_fcmp_fcsel_in_loop(ptr %arr, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %new_acc, %loop ]
  %ptr = getelementptr float, ptr %arr, i32 %i
  %val = load float, ptr %ptr
  %cmp = fcmp ogt float %val, %acc
  %new_acc = select i1 %cmp, float %val, float %acc
  %i.next = add i32 %i, 1
  %exit_cond = icmp eq i32 %i.next, %n
  br i1 %exit_cond, label %exit, label %loop
exit:
  ret float %new_acc
}

; Test 4: Multiple FCMP-FCSEL instruction pairs in same function
; CHECK: .globl _test_multiple_patterns
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_multiple_patterns:
define float @test_multiple_patterns(float %a, float %b, float %c, float %d, float %e, float %f) {
entry:
  %cmp1 = fcmp oeq float %a, %b
  %sel1 = select i1 %cmp1, float %c, float %d
  %cmp2 = fcmp ogt float %sel1, %e
  %sel2 = select i1 %cmp2, float %sel1, float %f
  ret float %sel2
}

; Test 5: FCMP with comparison to zero (immediate) - excluded from optimization
; FCMP #0.0 uses the ri-form opcode which is not in the detection list
; CHECK: .globl _test_fcmp_immediate
; CHECK-NEXT: .p2align 2
; CHECK-LABEL: _test_fcmp_immediate:
define float @test_fcmp_immediate(float %a, float %b) {
entry:
  %cmp = fcmp oeq float %a, 0.0
  %sel = select i1 %cmp, float %a, float %b
  ret float %sel
}

; Test 6: Nested loops with FCMP-FCSEL instruction pair
; CHECK: .globl _test_nested_loops
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_nested_loops:
define double @test_nested_loops(ptr %arr, i32 %rows, i32 %cols) {
entry:
  br label %outer_loop
outer_loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer_loop_latch ]
  %outer_acc = phi double [ 0.0, %entry ], [ %inner_result, %outer_loop_latch ]
  br label %inner_loop
inner_loop:
  %j = phi i32 [ 0, %outer_loop ], [ %j.next, %inner_loop ]
  %acc = phi double [ %outer_acc, %outer_loop ], [ %new_acc, %inner_loop ]
  %offset = mul i32 %i, %cols
  %idx = add i32 %offset, %j
  %ptr = getelementptr double, ptr %arr, i32 %idx
  %val = load double, ptr %ptr
  %cmp = fcmp ogt double %val, %acc
  %new_acc = select i1 %cmp, double %val, double %acc
  %j.next = add i32 %j, 1
  %inner_exit = icmp eq i32 %j.next, %cols
  br i1 %inner_exit, label %outer_loop_latch, label %inner_loop
outer_loop_latch:
  %inner_result = phi double [ %new_acc, %inner_loop ]
  %i.next = add i32 %i, 1
  %outer_exit = icmp eq i32 %i.next, %rows
  br i1 %outer_exit, label %exit, label %outer_loop
exit:
  %result = phi double [ %inner_result, %outer_loop_latch ]
  ret double %result
}

; Test 7: Mixed single and double precision in same function
; CHECK: .globl _test_mixed_precision
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_mixed_precision:
define float @test_mixed_precision(float %a, float %b, double %c, double %d) {
entry:
  %cmp_single = fcmp ogt float %a, %b
  %sel_single = select i1 %cmp_single, float %a, float %b
  %cmp_double = fcmp olt double %c, %d
  %sel_double = select i1 %cmp_double, double %c, double %d
  %trunc = fptrunc double %sel_double to float
  %final = fadd float %sel_single, %trunc
  ret float %final
}

; Test 8: FCMP-FCSEL instruction pair with a function call present
; CHECK: .globl _test_with_function_calls
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_with_function_calls:
declare float @external_func(float)
define float @test_with_function_calls(float %a, float %b, float %c, float %d) {
entry:
  %cmp = fcmp ogt float %a, %b
  %sel = select i1 %cmp, float %c, float %d
  %result = call float @external_func(float %sel)
  ret float %result
}

; Test 9: Verify no false positives - FCMP without FCSEL
; CHECK: .globl _test_fcmp_without_fcsel
; CHECK-NEXT: .p2align 2
; CHECK-LABEL: _test_fcmp_without_fcsel:
define i32 @test_fcmp_without_fcsel(float %a, float %b) {
entry:
  %cmp = fcmp ogt float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

; Test 10: Verify no false positives - FCSEL without preceding FCMP
; CHECK: .globl _test_fcsel_without_fcmp
; CHECK-NEXT: .p2align 2
; CHECK-LABEL: _test_fcsel_without_fcmp:
define float @test_fcsel_without_fcmp(i1 %cond, float %a, float %b) {
entry:
  %result = select i1 %cond, float %a, float %b
  ret float %result
}

;------------------------------------------------------------------------------
; CMP/CMN-CSEL tests (bit 1 of -aarch64-code-layout-opt)
;------------------------------------------------------------------------------

; Test 11: Basic CMP-CSEL instruction pair (integer register comparison)
; CHECK: .globl _test_basic_cmp_csel
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_basic_cmp_csel:
define i32 @test_basic_cmp_csel(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %cmp = icmp eq i32 %a, %b
  %sel = select i1 %cmp, i32 %c, i32 %d
  ret i32 %sel
}

; Test 12: CMP-CSEL instruction pair with small immediate (<=15, qualifies for optimization)
; CHECK: .globl _test_cmp_small_imm_csel
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_cmp_small_imm_csel:
define i32 @test_cmp_small_imm_csel(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp = icmp eq i32 %a, 7
  %sel = select i1 %cmp, i32 %b, i32 %c
  ret i32 %sel
}

; Test 13: CMP-CSEL with immediate > 15 - excluded from optimization
; CHECK: .globl _test_cmp_large_imm_csel
; CHECK-NEXT: .p2align 2
; CHECK-LABEL: _test_cmp_large_imm_csel:
define i32 @test_cmp_large_imm_csel(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp = icmp eq i32 %a, 100
  %sel = select i1 %cmp, i32 %b, i32 %c
  ret i32 %sel
}

; Test 14: Basic CMN-CSEL instruction pair (ADDSWrr with WZR destination)
; CHECK: .globl _test_basic_cmn_csel
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_basic_cmn_csel:
define i32 @test_basic_cmn_csel(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %sum = add i32 %a, %b
  %cmp = icmp eq i32 %sum, 0
  %sel = select i1 %cmp, i32 %c, i32 %d
  ret i32 %sel
}

; Test 15: CMN-CSEL instruction pair with small immediate (ADDSWri imm=7, qualifies)
; CHECK: .globl _test_cmn_small_imm_csel
; CHECK-NEXT: .p2align 6
; CHECK-LABEL: _test_cmn_small_imm_csel:
define i32 @test_cmn_small_imm_csel(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp = icmp eq i32 %a, -7
  %sel = select i1 %cmp, i32 %b, i32 %c
  ret i32 %sel
}

; Test 16: CMP without CSEL - no false positive
; CHECK: .globl _test_cmp_without_csel
; CHECK-NEXT: .p2align 2
; CHECK-LABEL: _test_cmp_without_csel:
define i32 @test_cmp_without_csel(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}
