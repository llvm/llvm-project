; This used to incorrectly use a TMLL for an always-false test at -O0.
;
; RUN: llc -O0 < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @test(ptr %input, ptr %result) {
entry:
; CHECK-NOT: tmll

  %0 = load i8, ptr %input, align 1
  %1 = trunc i8 %0 to i1
  %2 = zext i1 %1 to i32
  %3 = icmp sge i32 %2, 0
  br i1 %3, label %if.then, label %if.else

if.then:
  store i32 1, ptr %result, align 4
  br label %return

if.else:
  store i32 0, ptr %result, align 4
  br label %return

return:
  ret void
}

