; This produced masked gather that we are not yet handling
; REQUIRES: asserts
; RUN: opt -march=hexagon -passes=loop-vectorize -hexagon-autohvx -mattr=+hvx-length128b,+hvxv68,+v68,+hvx-ieee-fp,-long-calls,-packets -debug-only=loop-vectorize -S < %s 2>&1 | FileCheck %s

; Original C++
; clang -c -Os -mhvx -mhvx-ieee-fp -fvectorize -mno-packets -fno-strict-aliasing -Os -mhvx -mhvx-ieee-fp  -mno-packets -mv68
;typedef struct poptContext_s * poptContext;
;typedef struct { unsigned int bits[1]; } pbm_set;
;struct poptContext_s { pbm_set * arg_strip; };
;
;int poptStrippedArgv(poptContext con, int argc, char ** argv) {
;  int numargs = argc;
;   for (int i = 1; i < argc; i++) {
;     if (((((con->arg_strip)->bits)[((i) / (8 * sizeof (unsigned int)))] & ((unsigned int) 1 << ((i) % (8 * sizeof (unsigned int))))) != 0))
;     numargs--;
;   }
;    return numargs;
;}

; CHECK-NOT: masked_gather

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-unknown-elf"

; Function Attrs: nofree norecurse nosync nounwind optsize memory(read, inaccessiblemem: none)
define dso_local i32 @poptStrippedArgv(ptr noundef readonly captures(none) %con, i32 noundef %argc, ptr noundef readnone captures(none) %argv) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp sgt i32 %argc, 1
  br i1 %cmp8, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %0 = load ptr, ptr %con, align 4
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %spec.select.lcssa = phi i32 [ %spec.select, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %numargs.0.lcssa = phi i32 [ %argc, %entry ], [ %spec.select.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %numargs.0.lcssa

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.010 = phi i32 [ 1, %for.body.lr.ph ], [ %inc, %for.body ]
  %numargs.09 = phi i32 [ %argc, %for.body.lr.ph ], [ %spec.select, %for.body ]
  %div7 = lshr i32 %i.010, 5
  %arrayidx = getelementptr inbounds nuw [1 x i32], ptr %0, i32 0, i32 %div7
  %1 = load i32, ptr %arrayidx, align 4
  %rem = and i32 %i.010, 31
  %shl = shl nuw i32 1, %rem
  %and = and i32 %1, %shl
  %cmp1.not = icmp ne i32 %and, 0
  %dec = sext i1 %cmp1.not to i32
  %spec.select = add nsw i32 %numargs.09, %dec
  %inc = add nuw nsw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %argc
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
