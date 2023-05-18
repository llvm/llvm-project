; RUN: llc < %s -verify-machineinstrs -mtriple=i686-linux -mattr=-sse | FileCheck %s
; PR11768

@ptr = external dso_local global ptr

define void @baz() nounwind ssp {
entry:
  %0 = load ptr, ptr @ptr, align 4
  %cmp = icmp eq ptr %0, null
  fence seq_cst
  br i1 %cmp, label %if.then, label %if.else

; Make sure the fence comes before the comparison, since it
; clobbers EFLAGS.

; CHECK: lock orl {{.*}}, (%esp)
; CHECK-NEXT: testl [[REG:%e[a-z]+]], [[REG]]

if.then:                                          ; preds = %entry
  tail call void @foo() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare void @foo(...)

declare void @bar(...)
