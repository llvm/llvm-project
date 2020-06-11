; RUN: llc < %s -asm-verbose=false -verify-machineinstrs | FileCheck %s

;; Test that a small but nontrivial switch in a loop (like in a
;; bytecode interpreter) lowers reasonably without any irreducible
;; control flow being introduced.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32"

declare void @a(i32*)
declare void @b(i32*)

; CHECK-LABEL: switch_in_loop:
; CHECK-NEXT: .functype switch_in_loop (i32, i32) -> (i32)
; CHECK:    global.get __stack_pointer
; CHECK:    global.set __stack_pointer
; CHECK:    block
; CHECK:    br_if 0
; CHECK: .LBB0_2:
; CHECK:    loop
; CHECK:    block
; CHECK:    block
; CHECK:    block
; CHECK:    br_table {0, 1, 2}
; CHECK: .LBB0_3:
; CHECK:    end_block
; CHECK:    call a
; CHECK:    br 1
; CHECK: .LBB0_4:
; CHECK:    end_block
; CHECK:    call b
; CHECK: .LBB0_5:
; CHECK:    end_block
; CHECK:    br_if 0
; CHECK:    end_loop
; CHECK: .LBB0_7:
; CHECK:    end_block
; CHECK:    global.set __stack_pointer
; CHECK:    end_function
define i32 @switch_in_loop(i32* %ops, i32 %len) {
entry:
  %res = alloca i32
  %0 = bitcast i32* %res to i8*
  store i32 0, i32* %res
  %cmp6 = icmp sgt i32 %len, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.cond.cleanup.loopexit:                        ; preds = %sw.epilog
  %.pre = load i32, i32* %res
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %1 = phi i32 [ %.pre, %for.cond.cleanup.loopexit ], [ 0, %entry ]
  ret i32 %1

for.body:                                         ; preds = %entry, %sw.epilog
  %i.07 = phi i32 [ %inc, %sw.epilog ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %ops, i32 %i.07
  %2 = load i32, i32* %arrayidx
  switch i32 %2, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
  ]

sw.bb:                                            ; preds = %for.body
  call void @a(i32* nonnull %res)
  br label %sw.epilog

sw.bb1:                                           ; preds = %for.body
  call void @b(i32* nonnull %res)
  br label %sw.epilog

sw.epilog:                                        ; preds = %for.body, %sw.bb1, %sw.bb
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %len
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
