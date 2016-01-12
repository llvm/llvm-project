; RUN: llc <%s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%struct.buffer_t = type { i64, i8*, [4 x i32], [4 x i32], [4 x i32], i32, i8, i8, [2 x i8] }

declare i32 @__f1(i8*, %struct.buffer_t* noalias)

; CHECK-LABEL: f1:
define i32 @f1(i8* %__user_context, %struct.buffer_t* noalias %f1.buffer) {
entry:
  br i1 undef, label %"assert succeeded", label %"assert failed", !prof !1

"assert failed":                                  ; preds = %entry
  br label %destructor_block

"assert succeeded":                               ; preds = %entry
  %__f1_result = call i32 @__f1(i8* %__user_context, %struct.buffer_t* %f1.buffer) #5
  %0 = icmp eq i32 %__f1_result, 0
  br i1 %0, label %"assert succeeded11", label %"assert failed10", !prof !1

destructor_block:                                 ; preds = %"assert succeeded11", %"assert failed10", %"assert failed"
  %1 = phi i32 [ undef, %"assert failed" ], [ %__f1_result, %"assert failed10" ], [ 0, %"assert succeeded11" ]
  ret i32 %1

"assert failed10":                                ; preds = %"assert succeeded"
  br label %destructor_block

"assert succeeded11":                             ; preds = %"assert succeeded"
  br label %destructor_block
}

attributes #5 = { nounwind }

!1 = !{!"branch_weights", i32 1073741824, i32 0}
