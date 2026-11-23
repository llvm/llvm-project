; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic | FileCheck %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic -stop-after=finalize-isel | \
; RUN:   FileCheck %s --check-prefix=MIR

;; The GOTDATA_OP load of a global's address reads the GOT, which is constant,
;; so it must carry a GOT memory operand and be hoisted out of the loop.

; MIR: GDOP_LDXrr {{.*}} @g :: (load (s64) from got)

; CHECK-LABEL: loop:
; CHECK:         ldx [{{%[gilo][0-7]}}+{{%[gilo][0-7]}}], {{%[gilo][0-7]}}, %gdop(g)
; CHECK:       .LBB0_{{[0-9]+}}:
; CHECK-NOT:     %gdop(g)
; CHECK:         call use
; CHECK-NOT:     %gdop(g)
; CHECK:       .Lfunc_end0:

@g = external global i32
declare void @use(i32)

define void @loop(i32 %n) {
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %v = load i32, ptr @g
  %c = icmp eq i32 %v, %i
  br i1 %c, label %call, label %latch
call:
  store i32 %i, ptr @g
  call void @use(i32 %i)
  br label %latch
latch:
  %inc = add i32 %i, 1
  %done = icmp eq i32 %inc, %n
  br i1 %done, label %exit, label %body
exit:
  ret void
}
