; RUN: opt < %s -basicaa -dse -enable-nonlocal-dse -S | FileCheck %s

; The add and store in entry block should be removed by non-local DSE.
; CHECK-NOT: %stval = add
; CHECK-NOT: store i32 %stval
;
define void @foo(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32 %c) {
entry:
  %0 = load i32, i32* %b, align 4
  %stval = add nsw i32 %0, 1
  store i32 %stval, i32* %a, align 4
  %cmp = icmp sgt i32 %c, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %1 = add nsw i32 %c, 10
  br label %if.end

if.else:
  %2 = add nsw i32 %c, 13
  br label %if.end

if.end:
  %3 = phi i32 [ %1, %if.then ], [ %2, %if.else ]
  store i32 %3, i32* %a, align 4
  ret void
}
