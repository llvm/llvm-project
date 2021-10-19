; RUN: opt < %s -unroll-runtime-other-exit-predictable=false -loop-unroll -unroll-runtime=true -verify-dom-info -verify-loop-info -instcombine -S | FileCheck %s
; RUN: opt < %s -unroll-runtime-other-exit-predictable=false -loop-unroll -unroll-runtime=true -verify-dom-info -unroll-runtime-multi-exit=false -verify-loop-info -S | FileCheck %s -check-prefix=NOUNROLL

; this tests when unrolling multiple exit loop occurs by default (i.e. without specifying -unroll-runtime-multi-exit)

; the second exit block is a deopt block. The loop has one exiting block other than the latch.
define i32 @test1(i32* nocapture %a, i64 %n) {
; CHECK-LABEL: test1(
; CHECK-LABEL:  header.epil:
; CHECK-NEXT:     %indvars.iv.epil = phi i64 [ %indvars.iv.next.epil, %latch.epil ], [ %indvars.iv.unr, %header.epil.preheader ]
; CHECK-LABEL:  otherexit.loopexit:
; CHECK-NEXT:     %sum.02.lcssa.ph = phi i32 [ %sum.02, %for.exiting_block ], [ %add, %for.exiting_block.1 ], [ %add.1, %for.exiting_block.2 ], [ %add.2, %for.exiting_block.3 ], [ %add.3, %for.exiting_block.4 ], [ %add.4, %for.exiting_block.5 ], [ %add.5, %for.exiting_block.6 ],
; CHECK-NEXT:     br label %otherexit
; CHECK-LABEL:  otherexit.loopexit3:
; CHECK-NEXT:     br label %otherexit
; CHECK-LABEL:  otherexit:
; CHECK-NEXT:     %sum.02.lcssa = phi i32 [ %sum.02.lcssa.ph, %otherexit.loopexit ], [ %sum.02.epil, %otherexit.loopexit3 ]
; CHECK-NEXT:     %rval = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %sum.02.lcssa) ]
; CHECK-NEXT:     ret i32 %rval
; CHECK-LABEL:  latch.7:
; CHECK:          add i64 %indvars.iv, 8

; NOUNROLL: test1(
; NOUNROLL-NOT: .epil
; NOUNROLL-NOT: .prol
; NOUNROLL:   otherexit:
; NOUNROLL-NEXT:   %sum.02.lcssa = phi i32 [ %sum.02, %for.exiting_block ]
; NOUNROLL-NEXT:   %rval = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %sum.02.lcssa) ] 
entry:
  br label %header

header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %latch ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %latch ], [ 0, %entry ]
  br label %for.exiting_block

for.exiting_block:
 %cmp = icmp eq i64 %n, 42
 br i1 %cmp, label %otherexit, label %latch

latch:
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %latchexit, label %header

latchexit:                                          ; preds = %latch
  %sum.0.lcssa = phi i32 [ %add, %latch ]
  ret i32 %sum.0.lcssa

otherexit:
  %rval = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %sum.02) ]
  ret i32 %rval
}

; the exit block is not a deopt block.
define i32 @test2(i32* nocapture %a, i64 %n) {
; CHECK-LABEL: test2(
; CHECK-NOT: .epil
; CHECK-NOT: .prol
; CHECK-LABEL: otherexit:
; CHECK-NEXT:    ret i32 %sum.02

entry:
  br label %header

header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %latch ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %latch ], [ 0, %entry ]
  br label %for.exiting_block

for.exiting_block:
 %cmp = icmp eq i64 %n, 42
 br i1 %cmp, label %otherexit, label %latch

latch:
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %latchexit, label %header

latchexit:                                          ; preds = %latch
  %sum.0.lcssa = phi i32 [ %add, %latch ]
  ret i32 %sum.0.lcssa

otherexit:
  %rval = phi i32 [%sum.02, %for.exiting_block ]
  ret i32 %rval
}
declare i32 @llvm.experimental.deoptimize.i32(...)
