; This used to incorrectly use a TMLL for an always-false test at -O0.
;
; RUN: llc -O0 < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @test(i8 *%input, i32 *%result) {
entry:
; CHECK-NOT: tmll

  %0 = load i8, i8* %input, align 1
  %1 = trunc i8 %0 to i1
  %2 = zext i1 %1 to i32
  %3 = icmp sge i32 %2, 0
  br i1 %3, label %if.then, label %if.else

if.then:
  store i32 1, i32* %result, align 4
  br label %return

if.else:
  store i32 0, i32* %result, align 4
  br label %return

return:
  ret void
}

