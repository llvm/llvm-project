; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=-1 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK: define {{.*}} @foo{{.*}}#[[outlined_func_attr:[0-9]+]]
define void @foo() noreturn cold {
  unreachable
}

; CHECK: define {{.*}} @bar.cold.1{{.*}}#[[outlined_func_attr]]
define void @bar() {
  br i1 undef, label %normal, label %exit

normal:
  unreachable

exit:
  ret void
}

@take_addr_of_foo = global ptr @foo
@take_addr_of_bar = global ptr @bar

; CHECK: attributes #[[outlined_func_attr]] = {
; CHECK-SAME: cold
; CHECK-SAME: minsize
