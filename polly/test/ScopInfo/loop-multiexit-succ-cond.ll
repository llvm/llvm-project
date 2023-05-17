; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s --check-prefix=IR
;
; The SCoP contains a loop with multiple exit blocks (BBs after leaving
; the loop). The current implementation of deriving their domain derives
; only a common domain for all of the exit blocks. We disabled loops with
; multiple exit blocks until this is fixed.
; XFAIL: *
;
; Check that we do not crash and generate valid IR.
;
; CHECK:      Assumed Context:
; CHECK-NEXT:   [count1, dobreak, count2] -> {  :  }
; CHECK-NEXT: Invalid Context:
; CHECK-NEXT:   [count1, dobreak, count2] -> {  : (count1 > 0 and dobreak > 0) or count1 <= 0 or (count1 > 0 and dobreak <= 0 and count2 > 0) }
;
; CHECK:      Stmt_loop_enter
; CHECK-NEXT:     Domain :=
; CHECK-NEXT:         [count1, dobreak, count2] -> { Stmt_loop_enter[] : count1 > 0 };

; CHECK:      Stmt_loop_break
; CHECK-NEXT:     Domain :=
; CHECK-NEXT:         [count1, dobreak, count2] -> { Stmt_loop_break[] : count1 > 0 and dobreak > 0 };

; CHECK:      Stmt_loop_finish
; CHECK-NEXT:     Domain :=
; CHECK-NEXT:         [count1, dobreak, count2] -> { Stmt_loop_finish[] : count1 > 0 and dobreak <= 0 and count2 > 0 };

; CHECK:      Stmt_loop_skip
; CHECK-NEXT:     Domain :=
; CHECK-NEXT:         [count1, dobreak, count2] -> { Stmt_loop_skip[] : count1 <= 0 };

; IR:      polly.merge_new_and_old:
; IR-NEXT:   %phi.ph.merge = phi float [ %phi.ph.final_reload, %polly.exiting ], [ %phi.ph, %return.region_exiting ]
; IR-NEXT:   br label %return
;
; IR:      return:
; IR-NEXT:   %phi = phi float [ %phi.ph.merge, %polly.merge_new_and_old ]

declare void @g();

define void @func(i64 %count1, i64 %count2, i32 %dobreak, ptr %A) {
entry:
  %fadd = fadd float undef, undef
  br label %loopguard

loopguard:
  %cmp6 = icmp sgt i64 %count1, 0
  br i1 %cmp6, label %loop_enter, label %loop_skip


loop_enter:
  store float 1.0, ptr %A
  br label %loop_header

loop_header:
  %indvars.iv63 = phi i64 [ %indvars.iv.next64, %loop_continue ], [ 0, %loop_enter ]
  %indvars.iv.next64 = add nuw nsw i64 %indvars.iv63, 1
  %add8 = add i64 undef, undef
  %cmp_break = icmp sge i32 %dobreak, 1
  br i1 %cmp_break, label %loop_break, label %loop_continue

loop_continue:
  %cmp9 = icmp eq i64 %indvars.iv.next64, %count2
  br i1 %cmp9, label %loop_finish, label %loop_header


loop_break:
  store float 2.0, ptr %A
  br label %loop_break_error

loop_break_error:
  %cmp_loop_break = fcmp oeq float %fadd, 2.
  br i1 %cmp_loop_break, label %loop_break_g, label %return

loop_break_g:
  call void @g()
  br label %return


loop_finish:
    store float 3.0, ptr %A
  br label %loop_finish_error

loop_finish_error:
  call void @g()
  br label %return


loop_skip:
  store float 4.0, ptr %A
  br label %loop_skip_error

loop_skip_error:
  call void @g()
  br label %return


return:
  %phi = phi float [ 0.0, %loop_finish_error ], [ 0.0, %loop_break_error ], [ 2.0, %loop_break_g ], [ 3.0, %loop_skip_error ]
  store float 1.0, ptr %A
  ret void
}
