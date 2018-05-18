; RUN: opt -S -basicaa -licm < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f() nounwind

; constant fold on first ieration
define i32 @test1(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test1(
entry:
; CHECK: %i1 = load i32, i32* %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %r.chk = icmp ult i32 %iv, 2000
  br i1 %r.chk, label %continue, label %fail
continue:
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; Same as test1, but with a floating point IR and fcmp
define i32 @test_fcmp(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test_fcmp(
entry:
; CHECK: %i1 = load i32, i32* %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body

for.body:
  %iv = phi float [ 0.0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %r.chk = fcmp olt float %iv, 2000.0
  br i1 %r.chk, label %continue, label %fail
continue:
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = fadd float %iv, 1.0
  %exitcond = fcmp ogt float %inc, 1000.0
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; Count down from a.length w/entry guard
; TODO: currently unable to prove the following:
; ule i32 (add nsw i32 %len, -1), %len where len is [0, 512]
define i32 @test2(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test2(
entry:
  %len = load i32, i32* %a, align 4, !range !{i32 0, i32 512}
  %is.non.pos = icmp eq i32 %len, 0
  br i1 %is.non.pos, label %fail, label %preheader
preheader:
  %lenminusone = add nsw i32 %len, -1
  br label %for.body
for.body:
  %iv = phi i32 [ %lenminusone, %preheader ], [ %dec, %continue ]
  %acc = phi i32 [ 0, %preheader ], [ %add, %continue ]
  %r.chk = icmp ule i32 %iv, %len
  br i1 %r.chk, label %continue, label %fail
continue:
; CHECK-LABEL: continue
; CHECK: %i1 = load i32, i32* %a, align 4
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %dec = add nsw i32 %iv, -1
  %exitcond = icmp eq i32 %dec, 0
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; trivially true for zero
define i32 @test3(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test3(
entry:
  %len = load i32, i32* %a, align 4, !range !{i32 0, i32 512}
  %is.zero = icmp eq i32 %len, 0
  br i1 %is.zero, label %fail, label %preheader
preheader:
; CHECK: %i1 = load i32, i32* %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body
for.body:
  %iv = phi i32 [ 0, %preheader ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %preheader ], [ %add, %continue ]
  %r.chk = icmp ule i32 %iv, %len
  br i1 %r.chk, label %continue, label %fail
continue:
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; requires fact length is non-zero
; TODO: IsKnownNonNullFromDominatingConditions is currently only be done for
; pointers; should handle integers too
define i32 @test4(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test4(
entry:
  %len = load i32, i32* %a, align 4, !range !{i32 0, i32 512}
  %is.zero = icmp eq i32 %len, 0
  br i1 %is.zero, label %fail, label %preheader
preheader:
  br label %for.body
for.body:
  %iv = phi i32 [ 0, %preheader ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %preheader ], [ %add, %continue ]
  %r.chk = icmp ult i32 %iv, %len
  br i1 %r.chk, label %continue, label %fail
continue:
; CHECK-LABEL: continue
; CHECK: %i1 = load i32, i32* %a, align 4
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; variation on test1 with branch swapped
define i32 @test-brswap(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test-brswap(
entry:
; CHECK: %i1 = load i32, i32* %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %r.chk = icmp ugt i32 %iv, 2000
  br i1 %r.chk, label %fail, label %continue
continue:
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

define i32 @test-nonphi(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test-nonphi(
entry:
  br label %for.body

for.body:
; CHECK-LABEL: continue
; CHECK: %i1 = load i32, i32* %a, align 4
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %xor = xor i32 %iv, 72
  %r.chk = icmp ugt i32 %xor, 2000
  br i1 %r.chk, label %fail, label %continue
continue:
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

define i32 @test-wrongphi(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test-wrongphi(
entry:
  br label %for.body
  
for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %cond = icmp ult i32 %iv, 500
  br i1 %cond, label %dummy_block1, label %dummy_block2

dummy_block1:
  br label %dummy_block2

dummy_block2:
  %wrongphi = phi i32 [11, %for.body], [12, %dummy_block1]
  %r.chk = icmp ugt i32 %wrongphi, 2000
  br i1 %r.chk, label %fail, label %continue
continue:
; CHECK-LABEL: continue
; CHECK: %i1 = load i32, i32* %a, align 4
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; This works because loop-simplify is run implicitly, but test for it anyways
define i32 @test-multiple-latch(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test-multiple-latch(
entry:
; CHECK: %i1 = load i32, i32* %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue1 ], [ %inc, %continue2 ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue1 ], [ %add, %continue2 ]
  %r.chk = icmp ult i32 %iv, 2000
  br i1 %r.chk, label %continue1, label %fail
continue1:
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %cmp = icmp eq i32 %add, 0
  br i1 %cmp, label %continue2, label %for.body
continue2:
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}
