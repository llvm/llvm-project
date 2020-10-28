; RUN: opt -loop-unswitch -enable-new-pm=0 %s -S | FileCheck %s

; When hoisting simple values out from a loop, and not being able to attempt to
; non-trivally unswitch the loop, due to the optsize attribute, the pass would
; return an incorrect Modified status. This was caught by the pass return
; status check that is hidden under EXPENSIVE_CHECKS.

; CHECK-LABEL: entry:
; CHECK-NEXT: %0 = call i32 @llvm.objectsize.i32.p0i8(i8* bitcast (%struct.anon* @b to i8*), i1 false, i1 false, i1 false)
; CHECK-NEXT: %1 = icmp uge i32 %0, 1
; CHECK-NEXT: br label %for.cond

%struct.anon = type { i16 }

@b = global %struct.anon zeroinitializer, align 1

; Function Attrs: minsize nounwind optsize
define i16 @c() #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %cont, %entry
  br label %for.inc

for.inc:                                          ; preds = %for.cond
  %0 = call i32 @llvm.objectsize.i32.p0i8(i8* bitcast (%struct.anon* @b to i8*), i1 false, i1 false, i1 false)
  %1 = icmp uge i32 %0, 1
  br i1 %1, label %cont, label %cont

cont:                                             ; preds = %for.inc
  %2 = load i16, i16* getelementptr inbounds (%struct.anon, %struct.anon* @b, i32 0, i32 0), align 1
  br label %for.cond
}

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.objectsize.i32.p0i8(i8*, i1 immarg, i1 immarg, i1 immarg) #1

attributes #0 = { minsize nounwind optsize }
attributes #1 = { nounwind readnone speculatable willreturn }
