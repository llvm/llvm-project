; RUN: opt < %s -S -loop-unroll -unroll-runtime -unroll-peel-multi-deopt-exit 2>&1 | FileCheck %s

; Make sure that we can peel even if deopt block is not in immidiate exit.

; CHECK-LABEL: @basic
; CHECK: br i1 %c, label %{{.*}}, label %side_exit
; CHECK: br i1 %{{.*}}, label %[[NEXT0:.*]], label %exit
; CHECK: [[NEXT0]]:
; CHECK: br i1 %c, label %{{.*}}, label %side_exit
; CHECK: br i1 %{{.*}}, label %[[NEXT1:.*]], label %exit

define i32 @basic(i32* %p, i32 %k, i1 %c) #0 !prof !1 {
entry:
  %cmp3 = icmp slt i32 0, %k
  br i1 %cmp3, label %header, label %deopt

header:
  br label %body

body:
  %i = phi i32 [ 0, %header ], [ %inc, %backedge ]
  %addr = getelementptr inbounds i32, i32* %p, i32 %i
  store i32 %i, i32* %addr, align 4
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %k
  br i1 %c, label %backedge, label %side_exit, !prof !3

backedge:
  br i1 %cmp, label %body, label %exit, !prof !2

exit:
  ret i32 %i

side_exit:
  br label %deopt

deopt:
  %deopt_kind = phi i32 [0, %entry], [1, %side_exit]
  %rval = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %deopt_kind) ]
  ret i32 %rval
}

declare i32 @llvm.experimental.deoptimize.i32(...)

!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 0, i32 1}
!3 = !{!"branch_weights", i32 1, i32 0}
