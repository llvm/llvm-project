; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; This test checks that the SimplifyCFG pass won't duplicate a call to a
; function marked noduplicate.
;
; CHECK-LABEL: @noduplicate
; CHECK: call void @barrier
; CHECK-NOT: call void @barrier
define void @noduplicate(i32 %cond, ptr %out) {
entry:
  %out1 = getelementptr i32, ptr %out, i32 1
  %out2 = getelementptr i32, ptr %out, i32 2
  %cmp = icmp eq i32 %cond, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 5, ptr %out
  br label %if.end

if.end:
  call void @barrier() #0
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:
  store i32 5, ptr %out1
  br label %cond.end

cond.end:
  %value = phi i32 [ 1, %cond.false ], [ 0, %if.end ]
  store i32 %value, ptr %out2
  ret void
}

; Function Attrs: noduplicate nounwind
declare void @barrier() #0

attributes #0 = { noduplicate nounwind }
