; RUN: opt -passes='loop(loop-reduce)' %s -o - -S | FileCheck %s

; Required metadata to trigger previously failing assertion.
target datalayout = "e-m:e-i64:64-n32:64"

@f = external dso_local local_unnamed_addr global i32, align 4

declare i32 @a() local_unnamed_addr
declare i32 @e(i32) local_unnamed_addr

define dso_local i32 @b() {
entry:
  %call = tail call i32 @a()
  %tobool.not = icmp eq i32 %call, 0
  br i1 %tobool.not, label %cleanup.cont.critedge, label %if.then

if.then:                                          ; preds = %entry
; It's ok to modify this test in the future should be able to split critical
; edges here, just noting that this is the critical edge that we care about.
; CHECK: callbr void asm sideeffect "", "X"(i8* blockaddress(@b, %cleanup.cont.critedge))
; CHECK-NEXT: to label %return [label %cleanup.cont.critedge]
  callbr void asm sideeffect "", "X"(i8* blockaddress(@b, %cleanup.cont.critedge))
          to label %return [label %cleanup.cont.critedge]

cleanup.cont.critedge:                            ; preds = %entry, %if.then
  br label %return

return:                                           ; preds = %if.then, %cleanup.cont.critedge
  %retval.0 = phi i32 [ 4, %cleanup.cont.critedge ], [ 0, %if.then ]
  ret i32 %retval.0
}

define dso_local i32 @do_pages_move_nr_pages() local_unnamed_addr {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end3, %entry
  %g.0 = phi i32 [ undef, %entry ], [ %inc, %if.end3 ]
  %0 = load i32, i32* @f, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end3, label %if.then

if.then:                                          ; preds = %for.cond
  %call.i = tail call i32 @a()
  %tobool.not.i = icmp eq i32 %call.i, 0
  br i1 %tobool.not.i, label %if.then2, label %if.then.i

if.then.i:                                        ; preds = %if.then
  callbr void asm sideeffect "", "X"(i8* blockaddress(@do_pages_move_nr_pages, %if.then2))
          to label %if.end3 [label %if.then2]

if.then2:                                         ; preds = %if.then, %if.then.i
  %g.0.lcssa = phi i32 [ %g.0, %if.then ], [ %g.0, %if.then.i ]
  %call4 = tail call i32 @e(i32 %g.0.lcssa)
  ret i32 undef

if.end3:                                          ; preds = %for.cond, %if.then.i
  %inc = add nsw i32 %g.0, 1
  br label %for.cond
}

