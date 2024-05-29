; RUN: opt -disable-output -debugify-each -passes=gvn < %s 2>&1 | FileCheck %s
; RUN: opt --experimental-debuginfo-iterators=false -disable-output -debugify-each -passes=gvn < %s 2>&1 | FileCheck %s

; CHECK-NOT: ERROR: Instruction with empty DebugLoc in function _Z3bazv --  {{%.*}} = phi
; CHECK: CheckFunctionDebugify [GVNPass]: PASS

@foo = dso_local local_unnamed_addr global i32 0, align 4
@x = global i8 17

define dso_local void @_Z3bazv() local_unnamed_addr #0 {
entry:
  br label %for.cond

for.cond.loopexit.loopexit:                       ; preds = %for.inc
  br label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.cond.loopexit.loopexit, %for.cond
  br label %for.cond

for.cond:                                         ; preds = %for.cond.loopexit, %entry
  %.pr = load i32, ptr @foo, align 4
  %tobool1 = icmp eq i32 %.pr, 0
  br i1 %tobool1, label %for.cond.loopexit, label %for.inc.preheader

for.inc.preheader:                                ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.inc.preheader, %for.inc
  %val = load i8, ptr @x
  %conv = sext i8 %val to i32
  store i32 %conv, ptr @foo, align 4
  %tobool = icmp eq i8 %val, 0
  br i1 %tobool, label %for.cond.loopexit.loopexit, label %for.inc
}

declare dso_local signext i8 @_Z3barv() local_unnamed_addr #1
