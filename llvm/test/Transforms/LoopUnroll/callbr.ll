; RUN: opt -loop-unroll -S %s | FileCheck %s

; Check that the loop body exists.
; CHECK: for.body
; CHECK: if.then
; CHECK: asm.fallthrough
; CHECK: l_yes
; CHECK: for.inc

; Check that the loop body does not get unrolled.  We could modify this test in
; the future to support loop unrolling callbr's IFF we checked that the callbr
; operands were unrolled/updated correctly, as today they are not.
; CHECK-NOT: if.then.1
; CHECK-NOT: asm.fallthrough.1
; CHECK-NOT: l_yes.1
; CHECK-NOT: for.inc.1
; CHECK-NOT: if.then.2
; CHECK-NOT: asm.fallthrough.2
; CHECK-NOT: l_yes.2
; CHECK-NOT: for.inc.2

define dso_local void @d() {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  ret void

for.body:                                         ; preds = %for.inc, %entry
  %e.04 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %tobool = icmp eq i32 %e.04, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  callbr void asm sideeffect "1: nop\0A\09.quad b, ${0:l}, $$5\0A\09", "!i,~{dirflag},~{fpsr},~{flags}"()
          to label %asm.fallthrough [label %l_yes]

asm.fallthrough:                                  ; preds = %if.then
  br label %l_yes

l_yes:                                            ; preds = %asm.fallthrough, %if.then
  %call = tail call i32 (...) @g()
  br label %for.inc

for.inc:                                          ; preds = %for.body, %l_yes
  %inc = add nuw nsw i32 %e.04, 1
  %exitcond = icmp eq i32 %inc, 3
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare dso_local i32 @g(...) local_unnamed_addr
