; RUN: llc -mtriple=powerpc64le-linux-gnu -mcpu=pwr8 < %s | FileCheck %s

; https://bugs.llvm.org/show_bug.cgi?id=32442
; Don't generate zero extension for the return value.
; CHECK-NOT: clrldi

define zeroext i1 @foo(i32 signext %i, ptr %p) {
entry:
  %cmp = icmp eq i32 %i, 0
  br i1 %cmp, label %return, label %if.end

if.end:
  store i32 %i, ptr %p, align 4
  br label %return

return:
  %retval = phi i1 [ true, %if.end ], [ false, %entry ]
  ret i1 %retval
}
