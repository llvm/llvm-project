; RUN: opt %loadNPMPolly '-passes=polly-custom<prepare;scops>' -polly-print-scops -tailcallopt -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NOLICM

;    void foo(int n, float A[static const restrict n], float x) {
;      //      (0)
;      for (int i = 0; i < 5; i += 1) {
;        for (int j = 0; j < n; j += 1) {
;          x = 7; // (1)
;        }
;        A[0] = x; // (3)
;      }
;      // (4)
;    }

; NOLICM: Statements

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i32 %n, ptr noalias nonnull %A, float %x) {
entry:
  %smax = call i32 @llvm.smax.i32(i32 %n, i32 0)
  %0 = add nuw i32 %smax, 1
  br label %for.cond.1.preheader

for.cond.1.preheader:                             ; preds = %entry, %for.end
  %i.05 = phi i32 [ 0, %entry ], [ %add5, %for.end ]
  %x.addr.04 = phi float [ %x, %entry ], [ %x.addr.1.lcssa, %for.end ]
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.cond.1, %for.cond.1.preheader
  %x.addr.1 = phi float [ 7.000000e+00, %for.cond.1 ], [ %x.addr.04, %for.cond.1.preheader ]
  %j.0 = phi i32 [ %add, %for.cond.1 ], [ 0, %for.cond.1.preheader ]
  %add = add nuw i32 %j.0, 1
  %exitcond = icmp ne i32 %add, %0
  br i1 %exitcond, label %for.cond.1, label %for.end

for.end:                                          ; preds = %for.cond.1
  %x.addr.1.lcssa = phi float [ %x.addr.1, %for.cond.1 ]
  store float %x.addr.1.lcssa, ptr %A, align 4
  %add5 = add nuw nsw i32 %i.05, 1
  %exitcond6 = icmp ne i32 %add5, 5
  br i1 %exitcond6, label %for.cond.1.preheader, label %for.end.6

for.end.6:                                        ; preds = %for.end
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }    
