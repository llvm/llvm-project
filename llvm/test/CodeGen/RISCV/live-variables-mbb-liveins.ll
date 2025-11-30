; Pre-regalloc test
; RUN: llc -mtriple=riscv64 -riscv-enable-live-variables -verify-machineinstrs \
; RUN: -riscv-liveness-update-kills -stop-after=riscv-live-variables \
; RUN: -riscv-liveness-update-mbb-liveins < %s | FileCheck %s

; Post-regalloc test
; RUN: llc -mtriple=riscv64 -riscv-enable-live-variables -verify-machineinstrs \
; RUN: -riscv-liveness-update-kills -stop-after=riscv-live-variables,1 \
; RUN: -riscv-liveness-update-mbb-liveins < %s | FileCheck %s --check-prefix=CHECK-PR

; Basic test: simple function with two arguments
; CHECK-LABEL: name: test_mbb_liveins
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11
; CHECK:   %1:gpr = COPY $x11
; CHECK:   %0:gpr = COPY $x10
; CHECK:   %2:gpr = ADD killed %0, killed %1
; CHECK:   $x10 = COPY killed %2
; CHECK:   PseudoRET implicit $x10

define i64 @test_mbb_liveins(i64 %a, i64 %b) {
entry:
  %sum = add i64 %a, %b
  ret i64 %sum
}

; Test with control flow: verify live-ins are correct for entry block with multiple args
; CHECK-PR-LABEL: name{{.*}}test_control_flow
; CHECK-PR: bb.0.entry:
; CHECK-PR:   successors:
; CHECK-PR:   liveins: $x10, $x11, $x12
; CHECK-PR:   BNE killed renamable $x12, killed $x0, %bb.2
; CHECK-PR:   PseudoBR %bb.1

; CHECK-PR: bb.1.then:
; CHECK-PR:   successors:
; CHECK-PR:   liveins: $x10, $x11
; CHECK-PR:   renamable $x10 = ADD killed renamable $x10, killed renamable $x11
; CHECK-PR:   PseudoBR %bb.3

; CHECK-PR: bb.2.else:
; CHECK-PR:   successors:
; CHECK-PR:   liveins: $x10, $x11
; CHECK-PR:   renamable $x10 = SUB killed renamable $x10, killed renamable $x11

; CHECK-PR: bb.3.end:
; CHECK-PR:   liveins: $x10
; CHECK-PR:   PseudoRET implicit killed $x10

define i64 @test_control_flow(i64 %a, i64 %b, i64 %cond) {
entry:
  %cmp = icmp eq i64 %cond, 0
  br i1 %cmp, label %then, label %else

then:
  %add = add i64 %a, %b
  br label %end

else:
  %sub = sub i64 %a, %b
  br label %end

end:
  %result = phi i64 [ %add, %then ], [ %sub, %else ]
  ret i64 %result
}

; Test with loops: verify live-ins for loop headers
; CHECK-LABEL: name: test_loop
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10
; CHECK: bb.1.loop:
; CHECK-NOT: liveins:
; CHECK: bb.2.exit:
; CHECK-NOT: liveins:

define i64 @test_loop(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %next = add i64 %i, 1
  %cmp = icmp ult i64 %next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i64 %next
}

; Test that reserved registers (like $x2 stack pointer) are NOT in live-ins
; This test uses function calls which implicitly use $x2 for stack operations
; CHECK-LABEL: name: test_reserved_regs
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11
; CHECK-NOT:   liveins: {{.*}}$x2
; CHECK:   ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:   $x10 = COPY killed %0
; CHECK:   $x11 = COPY killed %1
; CHECK:   PseudoCALL target-flags(riscv-call) @__adddi3

define i64 @test_reserved_regs(i64 %a, i64 %b) {
entry:
  %sum = call i64 @__adddi3(i64 %a, i64 %b)
  ret i64 %sum
}

declare i64 @__adddi3(i64, i64)

; Test with multiple arguments to verify all live-ins are tracked
; CHECK-LABEL: name: test_many_args
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11, $x12, $x13
; CHECK:   %3:gpr = COPY $x13
; CHECK:   %2:gpr = COPY $x12
; CHECK:   %1:gpr = COPY $x11
; CHECK:   %0:gpr = COPY $x10

define i64 @test_many_args(i64 %a, i64 %b, i64 %c, i64 %d) {
entry:
  %sum1 = add i64 %a, %b
  %sum2 = add i64 %c, %d
  %result = add i64 %sum1, %sum2
  ret i64 %result
}

; Test with nested control flow to ensure entry block has correct live-ins
; CHECK-LABEL: name: test_nested_control_flow
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11, $x12

define i64 @test_nested_control_flow(i64 %a, i64 %b, i64 %c) {
entry:
  %cmp1 = icmp eq i64 %c, 0
  br i1 %cmp1, label %outer_then, label %outer_else

outer_then:
  %cmp2 = icmp sgt i64 %a, %b
  br i1 %cmp2, label %inner_then, label %inner_else

inner_then:
  %add = add i64 %a, %b
  br label %end

inner_else:
  %sub = sub i64 %a, %b
  br label %end

outer_else:
  %mul = mul i64 %a, 2
  br label %end

end:
  %result = phi i64 [ %add, %inner_then ], [ %sub, %inner_else ], [ %mul, %outer_else ]
  ret i64 %result
}

; Test with doubly nested loops
; CHECK-LABEL: name: test_doubly_nested_loop
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11

define i64 @test_doubly_nested_loop(i64 %n, i64 %m) {
entry:
  br label %outer_loop

outer_loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %outer_latch ]
  %sum_outer = phi i64 [ 0, %entry ], [ %sum_final, %outer_latch ]
  %i_cmp = icmp ult i64 %i, %n
  br i1 %i_cmp, label %inner_loop, label %exit

inner_loop:
  %j = phi i64 [ 0, %outer_loop ], [ %j_next, %inner_loop ]
  %sum_inner = phi i64 [ %sum_outer, %outer_loop ], [ %sum_new, %inner_loop ]
  %sum_new = add i64 %sum_inner, %j
  %j_next = add i64 %j, 1
  %j_cmp = icmp ult i64 %j_next, %m
  br i1 %j_cmp, label %inner_loop, label %outer_latch

outer_latch:
  %sum_final = phi i64 [ %sum_new, %inner_loop ]
  %i_next = add i64 %i, 1
  br label %outer_loop

exit:
  ret i64 %sum_outer
}

; Test with triply nested loops and function calls
; This tests that reserved registers ($x2) are not added to live-ins even with deep nesting
; CHECK-LABEL: name: test_triply_nested_loop_with_calls
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11, $x12
; CHECK-NOT:   liveins: {{.*}}$x2

define i64 @test_triply_nested_loop_with_calls(i64 %n, i64 %m, i64 %p) {
entry:
  br label %loop_i

loop_i:
  %i = phi i64 [ 0, %entry ], [ %i_next, %loop_i_latch ]
  %sum_i = phi i64 [ 0, %entry ], [ %sum_after_j, %loop_i_latch ]
  %i_cmp = icmp ult i64 %i, %n
  br i1 %i_cmp, label %loop_j_header, label %exit

loop_j_header:
  br label %loop_j

loop_j:
  %j = phi i64 [ 0, %loop_j_header ], [ %j_next, %loop_j_latch ]
  %sum_j = phi i64 [ %sum_i, %loop_j_header ], [ %sum_after_k, %loop_j_latch ]
  %j_cmp = icmp ult i64 %j, %m
  br i1 %j_cmp, label %loop_k_header, label %loop_j_exit

loop_k_header:
  br label %loop_k

loop_k:
  %k = phi i64 [ 0, %loop_k_header ], [ %k_next, %loop_k ]
  %sum_k = phi i64 [ %sum_j, %loop_k_header ], [ %sum_k_new, %loop_k ]

  ; Function call inside innermost loop - uses stack pointer $x2 implicitly
  %prod = call i64 @__muldi3(i64 %i, i64 %j)
  %sum_k_new = add i64 %sum_k, %prod

  %k_next = add i64 %k, 1
  %k_cmp = icmp ult i64 %k_next, %p
  br i1 %k_cmp, label %loop_k, label %loop_k_exit

loop_k_exit:
  br label %loop_j_latch

loop_j_latch:
  %sum_after_k = phi i64 [ %sum_k_new, %loop_k_exit ]
  %j_next = add i64 %j, 1
  br label %loop_j

loop_j_exit:
  br label %loop_i_latch

loop_i_latch:
  %sum_after_j = phi i64 [ %sum_j, %loop_j_exit ]
  %i_next = add i64 %i, 1
  br label %loop_i

exit:
  ret i64 %sum_i
}

declare i64 @__muldi3(i64, i64)

; Test with unstructured control flow (multiple entries, irreducible loop)
; CHECK-LABEL: name: test_unstructured_cfg
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11, $x12

define i64 @test_unstructured_cfg(i64 %a, i64 %b, i64 %selector) {
entry:
  %cmp1 = icmp eq i64 %selector, 0
  br i1 %cmp1, label %block_a, label %block_b

block_a:
  %val_a = add i64 %a, 1
  %cmp_a = icmp slt i64 %val_a, 10
  br i1 %cmp_a, label %block_b, label %block_c

block_b:
  %phi_b = phi i64 [ %b, %entry ], [ %val_a, %block_a ], [ %val_c, %block_c ]
  %val_b = mul i64 %phi_b, 2
  %cmp_b = icmp sgt i64 %val_b, 100
  br i1 %cmp_b, label %exit, label %block_c

block_c:
  %phi_c = phi i64 [ %val_a, %block_a ], [ %val_b, %block_b ]
  %val_c = sub i64 %phi_c, 1
  %cmp_c = icmp ugt i64 %val_c, 5
  br i1 %cmp_c, label %block_b, label %block_a

exit:
  %result = phi i64 [ %val_b, %block_b ]
  ret i64 %result
}

; Test with multiple function calls and complex control flow
; Ensures reserved registers are not in live-ins across multiple call sites
; CHECK-LABEL: name: test_multiple_calls
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11, $x12, $x13
; CHECK-NOT:   liveins: {{.*}}$x2

define i64 @test_multiple_calls(i64 %a, i64 %b, i64 %c, i64 %d) {
entry:
  %cmp = icmp sgt i64 %a, %b
  br i1 %cmp, label %call1, label %call2

call1:
  %res1 = call i64 @__adddi3(i64 %a, i64 %b)
  %cmp1 = icmp eq i64 %res1, 0
  br i1 %cmp1, label %call3, label %merge

call2:
  %res2 = call i64 @__muldi3(i64 %c, i64 %d)
  br label %merge

call3:
  %res3 = call i64 @__adddi3(i64 %c, i64 %d)
  br label %merge

merge:
  %final = phi i64 [ %res1, %call1 ], [ %res2, %call2 ], [ %res3, %call3 ]
  ret i64 %final
}

; Test with switch-like control flow
; CHECK-LABEL: name: test_switch_cfg
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11

define i64 @test_switch_cfg(i64 %selector, i64 %value) {
entry:
  switch i64 %selector, label %default [
    i64 0, label %case0
    i64 1, label %case1
    i64 2, label %case2
    i64 3, label %case3
  ]

case0:
  %res0 = add i64 %value, 10
  br label %exit

case1:
  %res1 = sub i64 %value, 10
  br label %exit

case2:
  %res2 = mul i64 %value, 2
  br label %exit

case3:
  %res3 = shl i64 %value, 1
  br label %exit

default:
  br label %exit

exit:
  %result = phi i64 [ %res0, %case0 ], [ %res1, %case1 ], [ %res2, %case2 ], [ %res3, %case3 ], [ 0, %default ]
  ret i64 %result
}

; Test with loop with multiple exits
; CHECK-LABEL: name: test_loop_multiple_exits
; CHECK: bb.0.entry:
; CHECK:   liveins: $x10, $x11, $x12
; CHECK:   %10:gpr = COPY killed %11

; CHECK: bb.3.loop_continue:
; CHECK:   %4:gpr = ADD killed %3, %9
; CHECK:   %5:gpr = ADDI killed %2, 1
; CHECK:   %6:gpr = ADD killed %1, %0
; CHECK:   PseudoBR %bb.1

; CHECK: bb.4.exit1:
; CHECK:   $x10 = COPY killed %3
; CHECK:   PseudoRET implicit $x10

; CHECK: bb.5.exit2:
; CHECK:   $x10 = COPY killed %1
; CHECK:   PseudoRET implicit $x10

; CHECK-PR-LABEL: test_loop_multiple_exits
; CHECK-PR:  bb.0.entry:
; CHECK-PR:    successors:
; CHECK-PR:    liveins: $x10, $x11, $x12
; CHECK-PR:    renamable $x13 = COPY killed $x10
; CHECK-PR:    renamable $x10 = COPY $x0
; CHECK-PR:    renamable $x15 = COPY $x0
; CHECK-PR:    renamable $x14 = COPY killed $x0
; CHECK-PR:    renamable $x16 = SLLI renamable $x12, 1

; CHECK-PR:  bb.1.loop:
; CHECK-PR:    successors:
; CHECK-PR:    liveins: $x10, $x11, $x12, $x13, $x14, $x15, $x16
; CHECK-PR:    BLTU renamable $x11, renamable $x14, %bb.4
; CHECK-PR:    PseudoBR %bb.2

; CHECK-PR:  bb.2.check2:
; CHECK-PR:    successors:
; CHECK-PR:    liveins: $x10, $x11, $x12, $x13, $x14, $x15, $x16
; CHECK-PR:    BLTU renamable $x13, renamable $x15, %bb.5
; CHECK-PR:    PseudoBR %bb.3

; CHECK-PR:  bb.3.loop_continue:
; CHECK-PR:    successors:
; CHECK-PR:    liveins: $x10, $x11, $x12, $x13, $x14, $x15, $x16
; CHECK-PR:    renamable $x14 = ADD killed renamable $x14, renamable $x12
; CHECK-PR:    renamable $x15 = ADDI killed renamable $x15, 1
; CHECK-PR:    renamable $x10 = ADD killed renamable $x10, renamable $x16
; CHECK-PR:    PseudoBR %bb.1

; CHECK-PR:  bb.4.exit1:
; CHECK-PR:    liveins: $x14
; CHECK-PR:    $x10 = COPY killed renamable $x14
; CHECK-PR:    PseudoRET implicit killed $x10

; CHECK-PR:  bb.5.exit2:
; CHECK-PR:    liveins: $x10
; CHECK-PR:    PseudoRET implicit killed $x10

define i64 @test_loop_multiple_exits(i64 %n, i64 %threshold, i64 %increment) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %loop_continue ]
  %sum = phi i64 [ 0, %entry ], [ %sum_next, %loop_continue ]

  %cmp1 = icmp ugt i64 %sum, %threshold
  br i1 %cmp1, label %exit1, label %check2

check2:
  %cmp2 = icmp ugt i64 %i, %n
  br i1 %cmp2, label %exit2, label %loop_continue

loop_continue:
  %sum_next = add i64 %sum, %increment
  %i_next = add i64 %i, 1
  br label %loop

exit1:
  ret i64 %sum

exit2:
  %final = mul i64 %sum, 2
  ret i64 %final
}