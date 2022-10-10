; REQUIRES: asserts
; RUN: opt < %s -passes="loop(loop-interchange,loop-interchange)" -cache-line-size=8 -verify-dom-info -verify-loop-info \
; RUN:  -debug-only=loop-interchange 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@g_75 = external global i32, align 1
@g_78 = external global [6 x ptr], align 1

; Loop interchange as a loopnest pass should always construct the loop nest from
; the outermost loop. This test case runs loop interchange twice. In the loop pass
; manager, it might occur that after the first loop interchange transformation
; the original outermost loop becomes a inner loop hence the loop nest constructed
; afterwards for the second loop interchange pass turns out to be a loop list of size
; 2 and is not valid. This causes functional issues.
;
; Make sure we always construct the valid and correct loop nest at the beginning
; of execution of a loopnest pass.

; CHECK: Processing LoopList of size = 3
; CHECK: Processing LoopList of size = 3
define void @loopnest_01() {
entry:
  br label %for.cond5.preheader.i.i.i

for.cond5.preheader.i.i.i:                        ; preds = %for.end16.i.i.i, %entry
  %storemerge11.i.i.i = phi i32 [ 4, %entry ], [ %sub18.i.i.i, %for.end16.i.i.i ]
  br label %for.cond8.preheader.i.i.i

for.cond8.preheader.i.i.i:                        ; preds = %for.inc14.i.i.i, %for.cond5.preheader.i.i.i
  %l_105.18.i.i.i = phi i16 [ 0, %for.cond5.preheader.i.i.i ], [ %add15.i.i.i, %for.inc14.i.i.i ]
  br label %for.body10.i.i.i

for.body10.i.i.i:                                 ; preds = %for.body10.i.i.i, %for.cond8.preheader.i.i.i
  %storemerge56.i.i.i = phi i16 [ 5, %for.cond8.preheader.i.i.i ], [ %sub.i.i.i, %for.body10.i.i.i ]
  %arrayidx.i.i.i = getelementptr [6 x ptr], ptr @g_78, i16 0, i16 %storemerge56.i.i.i
  store ptr @g_75, ptr %arrayidx.i.i.i, align 1
  %sub.i.i.i = add nsw i16 %storemerge56.i.i.i, -1
  br i1 true, label %for.inc14.i.i.i, label %for.body10.i.i.i

for.inc14.i.i.i:                                  ; preds = %for.body10.i.i.i
  %add15.i.i.i = add nuw nsw i16 %l_105.18.i.i.i, 1
  %exitcond.not.i.i.i = icmp eq i16 %add15.i.i.i, 6
  br i1 %exitcond.not.i.i.i, label %for.end16.i.i.i, label %for.cond8.preheader.i.i.i

for.end16.i.i.i:                                  ; preds = %for.inc14.i.i.i
  %sub18.i.i.i = add nsw i32 %storemerge11.i.i.i, -1
  %cmp.i10.not.i.i = icmp eq i32 %storemerge11.i.i.i, 0
  br i1 %cmp.i10.not.i.i, label %func_4.exit.i, label %for.cond5.preheader.i.i.i

func_4.exit.i:                                    ; preds = %for.end16.i.i.i
  unreachable
}
