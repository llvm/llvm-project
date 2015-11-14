; RUN: llc -mcpu=pwr7 -relocation-model=pic < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@x = thread_local global [1600 x i32] zeroinitializer, align 4

; Function Attrs: nounwind
define void @foo(i32 signext %v) #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %induction5 = or i64 %index, 1
  %0 = getelementptr inbounds [1600 x i32], [1600 x i32]* @x, i64 0, i64 %index
  %1 = getelementptr inbounds [1600 x i32], [1600 x i32]* @x, i64 0, i64 %induction5
  %2 = load i32, i32* %0, align 4
  %3 = load i32, i32* %1, align 4
  %4 = add nsw i32 %2, %v
  %5 = add nsw i32 %3, %v
  store i32 %4, i32* %0, align 4
  store i32 %5, i32* %1, align 4
  %index.next = add i64 %index, 2
  %6 = icmp eq i64 %index.next, 1600
  br i1 %6, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

; CHECK-LABEL: @foo
; CHECK-NOT: mtctr
; CHECK: __tls_get_addr

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"PIC Level", i32 2}

