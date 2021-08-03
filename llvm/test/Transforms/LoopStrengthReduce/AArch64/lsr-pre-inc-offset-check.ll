; RUN: llc -mtriple=aarch64-none-eabi -lsr-preferred-addressing-mode=preindexed %s -o - | FileCheck %s

; In LSR for constant offsets and steps, we can generate pre-inc
; accesses by having the offset equal the step and generate a reuse
; formula. However, there are cases where the step, results in an
; illegal addressing mode.

; In this test, we set the preferred addressing mode to be preindexed,
; in order to test a scenario where the step results in an illegal
; addressing mode and because of that it should not generate a reuse formula.

; This test was created in order to reproduce a bug that was observed
; when building a bootstrap build on an AArch64 machine, where the
; preferred addresing mode is preindexed.


%"Type" = type <{[166 x [338 x i8]]}>
define void @test_lsr_pre_inc_offset_check(%"Type"* %p) {
; CHECK-LABEL: test_lsr_pre_inc_offset_check:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    add x8, x0, #340
; CHECK-NEXT:    mov w9, #165
; CHECK-NEXT:    mov w10, #2
; CHECK-NEXT:  .LBB0_1: // %main
; CHECK-NEXT:    // =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    stur wzr, [x8, #-1]
; CHECK-NEXT:    strb w10, [x8]
; CHECK-NEXT:    subs x9, x9, #1
; CHECK-NEXT:    add x8, x8, #338
; CHECK-NEXT:    b.ne .LBB0_1
; CHECK-NEXT:  // %bb.2: // %exit
; CHECK-NEXT:    ret
entry:
  br label %main
exit:
  ret void
if.then:
  %arrayidx.i = getelementptr inbounds %"Type", %"Type"* %p, i64 0, i32 0, i64 %indvars, i64 1
  %0 = bitcast i8* %arrayidx.i to i32*
  store i32 0, i32* %0, align 1
  br label %if.end
if.end:
  %arrayidx.p = getelementptr inbounds %"Type", %"Type"* %p, i64 0, i32 0, i64 %indvars, i64 2
  store i8 2, i8* %arrayidx.p, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars, 1
  %add.i = add nuw i8 %begin, 1
  %cmp.i.not = icmp eq i64 %indvars.iv.next, 166
  br i1 %cmp.i.not, label %exit, label %main
main:
  %begin = phi i8 [ 1, %entry ], [ %add.i, %if.end ]
  %indvars = phi i64 [ 1, %entry ], [ %indvars.iv.next, %if.end ]
  br label %if.then
}
