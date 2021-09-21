; RUN: opt -lower-constant-intrinsics -S < %s | FileCheck %s

; This is a reproducer for a heap-use-after-free bug that occured due to trying
; to process a PHI node that was removed in a preceding worklist iteration. The
; conditional branch in %cont2.i will be replaced with an unconditional branch
; to %cont4.i. As a result of that, the PHI node in %handler.type_mismatch3.i
; will be left with one predecessor and will thus be removed in that iteration.

; CHECK-NOT: phi
; CHECK: cont2.i:
; CHECK-NEXT: br label %cont4.i
; CHECK-NOT: phi

%s = type { [2 x i16] }

define fastcc void @foo(%s* %p) unnamed_addr {
entry:
  %0 = bitcast %s* %p to i8*
  %1 = tail call i32 @llvm.objectsize.i32.p0i8(i8* %0, i1 false, i1 false, i1 false) #2
  %2 = icmp ne i32 %1, 0
  %.not1.i = icmp eq i32 %1, 0
  br label %for.cond

for.cond:                                         ; preds = %entry
  br label %cont.i

cont.i:                                           ; preds = %for.cond
  br i1 undef, label %cont2.i, label %cont2.thread.i

cont2.thread.i:                                   ; preds = %cont.i
  br label %handler.type_mismatch3.i

cont2.i:                                          ; preds = %cont.i
  br i1 %.not1.i, label %handler.type_mismatch3.i, label %cont4.i

handler.type_mismatch3.i:                         ; preds = %cont2.i, %cont2.thread.i
  %3 = phi i1 [ %2, %cont2.thread.i ], [ false, %cont2.i ]
  unreachable

cont4.i:                                          ; preds = %cont2.i
  unreachable
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.objectsize.i32.p0i8(i8*, i1 immarg, i1 immarg, i1 immarg) #1

attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nounwind }
