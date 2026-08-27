; RUN: opt %s -passes='loop(loop-versioning-licm)' -S | FileCheck %s
;
; LoopVersioningLICM must not refuse to version a loop solely because of a call
; that only accesses inaccessible memory. Such a call cannot alias any pointer
; accessed in the loop, so it is safe for versioning. This notably covers
; llvm.pseudoprobe, which is inserted on every block under sample-based
; profiling (in the presence of -fpseudo-probe-for-profiling), as well as other
; inaccessible-memory intrinsics such as llvm.sideeffect.

; A loop containing an llvm.pseudoprobe intrinsic is still versioned.
; CHECK-LABEL: @test_pseudoprobe_lvlicm(
; CHECK: lver.check
define double @test_pseudoprobe_lvlicm(ptr %x, ptr %y, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %ph, label %exit

ph:                                               ; preds = %entry
  br label %body

body:                                             ; preds = %body, %ph
  %i = phi i32 [ 0, %ph ], [ %inext, %body ]
  %sum = phi double [ 0.000000e+00, %ph ], [ %sumnext, %body ]
  %yidx = getelementptr inbounds double, ptr %y, i32 %i
  %yv = load double, ptr %yidx, align 8
  %add = fadd double %yv, 1.000000e+00
  %xidx = getelementptr inbounds double, ptr %x, i32 %i
  store double %add, ptr %xidx, align 8
  %x0 = load double, ptr %x, align 8              ; loop-invariant load
  %sumnext = fadd double %sum, %x0
  call void @llvm.pseudoprobe(i64 1234, i64 1, i32 0, i64 -1)
  %inext = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inext, %n
  br i1 %exitcond, label %exit, label %body

exit:                                             ; preds = %body, %entry
  %sumlcssa = phi double [ 0.000000e+00, %entry ], [ %sumnext, %body ]
  ret double %sumlcssa
}

; The same holds for any call that only accesses inaccessible memory, e.g.
; llvm.sideeffect.
; CHECK-LABEL: @test_sideeffect_lvlicm(
; CHECK: lver.check
define double @test_sideeffect_lvlicm(ptr %x, ptr %y, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %ph, label %exit

ph:                                               ; preds = %entry
  br label %body

body:                                             ; preds = %body, %ph
  %i = phi i32 [ 0, %ph ], [ %inext, %body ]
  %sum = phi double [ 0.000000e+00, %ph ], [ %sumnext, %body ]
  %yidx = getelementptr inbounds double, ptr %y, i32 %i
  %yv = load double, ptr %yidx, align 8
  %add = fadd double %yv, 1.000000e+00
  %xidx = getelementptr inbounds double, ptr %x, i32 %i
  store double %add, ptr %xidx, align 8
  %x0 = load double, ptr %x, align 8              ; loop-invariant load
  %sumnext = fadd double %sum, %x0
  call void @llvm.sideeffect()
  %inext = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inext, %n
  br i1 %exitcond, label %exit, label %body

exit:                                             ; preds = %body, %entry
  %sumlcssa = phi double [ 0.000000e+00, %entry ], [ %sumnext, %body ]
  ret double %sumlcssa
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)
declare void @llvm.sideeffect()
