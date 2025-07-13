; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefixes=BFI_BEFORE
; RUN: opt < %s -passes='loop(loop-rotate),print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefixes=BFI_AFTER
; RUN: opt < %s -passes='loop(loop-rotate)' -S | FileCheck %s --check-prefixes=IR

@g = global i32 0

; We should get the same "count =" results for "outer_loop_body" and
; "inner_loop_body" before and after the transformation.

; BFI_BEFORE-LABEL: block-frequency-info: func0
; BFI_BEFORE: - entry: {{.*}} count = 1
; BFI_BEFORE: - outer_loop_header: {{.*}} count = 1001
; BFI_BEFORE: - outer_loop_body: {{.*}} count = 1000
; BFI_BEFORE: - inner_loop_header: {{.*}} count = 4000
; BFI_BEFORE: - inner_loop_body: {{.*}} count = 3000
; BFI_BEFORE: - inner_loop_exit: {{.*}} count = 1000
; BFI_BEFORE: - outer_loop_exit: {{.*}} count = 1

; BFI_AFTER-LABEL: block-frequency-info: func0
; BFI_AFTER: - entry: {{.*}} count = 1
; BFI_AFTER: - outer_loop_body: {{.*}} count = 1000
; BFI_AFTER: - inner_loop_body: {{.*}} count = 3000
; BFI_AFTER: - inner_loop_exit: {{.*}} count = 1000
; BFI_AFTER: - outer_loop_exit: {{.*}} count = 1

; IR-LABEL: define void @func0
; IR: inner_loop_body:
; IR:   br i1 %cmp1, label %inner_loop_body, label %inner_loop_exit, !prof [[PROF_FUNC0_0:![0-9]+]]
; IR: inner_loop_exit:
; IR:   br i1 %cmp0, label %outer_loop_body, label %outer_loop_exit, !prof [[PROF_FUNC0_1:![0-9]+]]
;
; A function with known loop-bounds where after loop-rotation we end with an
; unconditional branch in the pre-header.
define void @func0() !prof !0 {
entry:
  br label %outer_loop_header

outer_loop_header:
  %i0 = phi i32 [0, %entry], [%i0_inc, %inner_loop_exit]
  %cmp0 = icmp slt i32 %i0, 1000
  br i1 %cmp0, label %outer_loop_body, label %outer_loop_exit, !prof !1

outer_loop_body:
  store volatile i32 %i0, ptr @g, align 4
  br label %inner_loop_header

inner_loop_header:
  %i1 = phi i32 [0, %outer_loop_body], [%i1_inc, %inner_loop_body]
  %cmp1 = icmp slt i32 %i1, 3
  br i1 %cmp1, label %inner_loop_body, label %inner_loop_exit, !prof !2

inner_loop_body:
  store volatile i32 %i1, ptr @g, align 4
  %i1_inc = add i32 %i1, 1
  br label %inner_loop_header

inner_loop_exit:
  %i0_inc = add i32 %i0, 1
  br label %outer_loop_header

outer_loop_exit:
  ret void
}

; BFI_BEFORE-LABEL: block-frequency-info: func1
; BFI_BEFORE: - entry: {{.*}} count = 1024
; BFI_BEFORE: - loop_header: {{.*}} count = 21504
; BFI_BEFORE: - loop_body: {{.*}} count = 20480
; BFI_BEFORE: - loop_exit: {{.*}} count = 1024

; BFI_AFTER-LABEL: block-frequency-info: func1
; BFI_AFTER: - entry: {{.*}} count = 1024
; BFI_AFTER: - loop_body.lr.ph: {{.*}} count = 1016
; BFI_AFTER: - loop_body: {{.*}} count = 20480
; BFI_AFTER: - loop_header.loop_exit_crit_edge: {{.*}} count = 1016
; BFI_AFTER: - loop_exit: {{.*}} count = 1024

; IR-LABEL: define void @func1
; IR: entry:
; IR:   br i1 %cmp1, label %loop_body.lr.ph, label %loop_exit, !prof [[PROF_FUNC1_0:![0-9]+]]

; IR: loop_body:
; IR:   br i1 %cmp, label %loop_body, label %loop_header.loop_exit_crit_edge, !prof [[PROF_FUNC1_1:![0-9]+]]

; A function with unknown loop-bounds so loop-rotation ends up with a
; condition jump in pre-header and loop body. branch_weight shows body is
; executed more often than header.
define void @func1(i32 %n) !prof !3 {
entry:
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_inc, %loop_body]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_body, label %loop_exit, !prof !4

loop_body:
  store volatile i32 %i, ptr @g, align 4
  %i_inc = add i32 %i, 1
  br label %loop_header

loop_exit:
  ret void
}

; BFI_BEFORE-LABEL: block-frequency-info: func2
; BFI_BEFORE: - entry: {{.*}} count = 1024
; BFI_BEFORE: - loop_header: {{.*}} count = 1056
; BFI_BEFORE: - loop_body: {{.*}} count = 32
; BFI_BEFORE: - loop_exit: {{.*}} count = 1024

; BFI_AFTER-LABEL: block-frequency-info: func2
; - entry: {{.*}} count = 1024
; - loop_body.lr.ph: {{.*}} count = 32
; - loop_body: {{.*}} count = 32
; - loop_header.loop_exit_crit_edge: {{.*}} count = 32
; - loop_exit: {{.*}} count = 1024

; IR-LABEL: define void @func2
; IR: entry:
; IR:   br i1 %cmp1, label %loop_exit, label %loop_body.lr.ph, !prof [[PROF_FUNC2_0:![0-9]+]]

; IR: loop_body:
; IR:   br i1 %cmp, label %loop_header.loop_exit_crit_edge, label %loop_body, !prof [[PROF_FUNC2_1:![0-9]+]]

; A function with unknown loop-bounds so loop-rotation ends up with a
; condition jump in pre-header and loop body. Similar to `func1` but here
; loop-exit count is higher than backedge count.
define void @func2(i32 %n) !prof !3 {
entry:
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_inc, %loop_body]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_exit, label %loop_body, !prof !5

loop_body:
  store volatile i32 %i, ptr @g, align 4
  %i_inc = add i32 %i, 1
  br label %loop_header

loop_exit:
  ret void
}

; BFI_BEFORE-LABEL: block-frequency-info: func3_zero_branch_weight
; BFI_BEFORE: - entry: {{.*}} count = 1024
; BFI_BEFORE: - loop_header: {{.*}} count = 2199023255552
; BFI_BEFORE: - loop_body: {{.*}} count = 2199023254528
; BFI_BEFORE: - loop_exit: {{.*}} count = 1024

; BFI_AFTER-LABEL: block-frequency-info: func3_zero_branch_weight
; BFI_AFTER: - entry: {{.*}} count = 1024
; BFI_AFTER: - loop_body.lr.ph: {{.*}} count = 1024
; BFI_AFTER: - loop_body: {{.*}} count = 2199023255552
; BFI_AFTER: - loop_header.loop_exit_crit_edge: {{.*}} count = 1024
; BFI_AFTER: - loop_exit: {{.*}} count = 1024

; IR-LABEL: define void @func3_zero_branch_weight
; IR: entry:
; IR:   br i1 %cmp1, label %loop_exit, label %loop_body.lr.ph, !prof [[PROF_FUNC3_0:![0-9]+]]

; IR: loop_body:
; IR:   br i1 %cmp, label %loop_header.loop_exit_crit_edge, label %loop_body, !prof [[PROF_FUNC3_0]]

define void @func3_zero_branch_weight(i32 %n) !prof !3 {
entry:
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_inc, %loop_body]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_exit, label %loop_body, !prof !6

loop_body:
  store volatile i32 %i, ptr @g, align 4
  %i_inc = add i32 %i, 1
  br label %loop_header

loop_exit:
  ret void
}

; IR-LABEL: define void @func4_zero_branch_weight
; IR: entry:
; IR:   br i1 %cmp1, label %loop_exit, label %loop_body.lr.ph, !prof [[PROF_FUNC4_0:![0-9]+]]

; IR: loop_body:
; IR:   br i1 %cmp, label %loop_header.loop_exit_crit_edge, label %loop_body, !prof [[PROF_FUNC4_0]]

define void @func4_zero_branch_weight(i32 %n) !prof !3 {
entry:
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_inc, %loop_body]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_exit, label %loop_body, !prof !7

loop_body:
  store volatile i32 %i, ptr @g, align 4
  %i_inc = add i32 %i, 1
  br label %loop_header

loop_exit:
  ret void
}

; IR-LABEL: define void @func5_zero_branch_weight
; IR: entry:
; IR:   br i1 %cmp1, label %loop_exit, label %loop_body.lr.ph, !prof [[PROF_FUNC5_0:![0-9]+]]

; IR: loop_body:
; IR:   br i1 %cmp, label %loop_header.loop_exit_crit_edge, label %loop_body, !prof [[PROF_FUNC5_0]]

define void @func5_zero_branch_weight(i32 %n) !prof !3 {
entry:
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_inc, %loop_body]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_exit, label %loop_body, !prof !8

loop_body:
  store volatile i32 %i, ptr @g, align 4
  %i_inc = add i32 %i, 1
  br label %loop_header

loop_exit:
  ret void
}

; BFI_BEFORE-LABEL: block-frequency-info: func6_inaccurate_branch_weight
; BFI_BEFORE: - entry: {{.*}} count = 1024
; BFI_BEFORE: - loop_header: {{.*}} count = 2047
; BFI_BEFORE: - loop_body: {{.*}} count = 1023
; BFI_BEFORE: - loop_exit: {{.*}} count = 1024

; BFI_AFTER-LABEL: block-frequency-info: func6_inaccurate_branch_weight
; BFI_AFTER: - entry: {{.*}} count = 1024
; BFI_AFTER: - loop_body: {{.*}} count = 1024
; BFI_AFTER: - loop_exit: {{.*}} count = 1024

; IR-LABEL: define void @func6_inaccurate_branch_weight(
; IR: entry:
; IR:   br label %loop_body
; IR: loop_body:
; IR:   br i1 %cmp, label %loop_body, label %loop_exit, !prof [[PROF_FUNC6_0:![0-9]+]]
; IR: loop_exit:
; IR:   ret void

; Branch weight from sample-based PGO may be inaccurate due to sampling.
; Count for loop_body in following case should be not less than loop_exit.
; However this may not hold for Sample-based PGO.
define void @func6_inaccurate_branch_weight() !prof !3 {
entry:
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_inc, %loop_body]
  %cmp = icmp slt i32 %i, 2
  br i1 %cmp, label %loop_body, label %loop_exit, !prof !9

loop_body:
  store volatile i32 %i, ptr @g, align 4
  %i_inc = add i32 %i, 1
  br label %loop_header

loop_exit:
  ret void
}

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 1000, i32 1}
!2 = !{!"branch_weights", i32 3000, i32 1000}
!3 = !{!"function_entry_count", i64 1024}
!4 = !{!"branch_weights", i32 40, i32 2}
!5 = !{!"branch_weights", i32 10240, i32 320}
!6 = !{!"branch_weights", i32 0, i32 1}
!7 = !{!"branch_weights", i32 1, i32 0}
!8 = !{!"branch_weights", i32 0, i32 0}
!9 = !{!"branch_weights", i32 1023, i32 1024}

; IR: [[PROF_FUNC0_0]] = !{!"branch_weights", i32 2000, i32 1000}
; IR: [[PROF_FUNC0_1]] = !{!"branch_weights", i32 999, i32 1}
; IR: [[PROF_FUNC1_0]] = !{!"branch_weights", i32 127, i32 1}
; IR: [[PROF_FUNC1_1]] = !{!"branch_weights", i32 2433, i32 127}
; IR: [[PROF_FUNC2_0]] = !{!"branch_weights", i32 9920, i32 320}
; IR: [[PROF_FUNC2_1]] = !{!"branch_weights", i32 320, i32 0}
; IR: [[PROF_FUNC3_0]] = !{!"branch_weights", i32 0, i32 1}
; IR: [[PROF_FUNC4_0]] = !{!"branch_weights", i32 1, i32 0}
; IR: [[PROF_FUNC5_0]] = !{!"branch_weights", i32 0, i32 0}
; IR: [[PROF_FUNC6_0]] = !{!"branch_weights", i32 0, i32 1024}
