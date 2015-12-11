; RUN: opt < %s -basicaa -dse -enable-nonlocal-dse -S | FileCheck %s

; The store and add in if.then block should be removed by non-local DSE.
; CHECK-NOT: %stval = add
; CHECK-NOT: store i32 %stval
;
define void @foo(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32 %c) {
entry:
  %cmp = icmp sgt i32 %c, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %0 = load i32, i32* %b, align 4
  %stval = add nsw i32 %0, 1
  store i32 %stval, i32* %a, align 4
  br label %if.end

if.end:
  %m.0 = phi i32 [ 13, %if.then ], [ 10, %entry ]
  store i32 %m.0, i32* %a, align 4
  ret void
}
