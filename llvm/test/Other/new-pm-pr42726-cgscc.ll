; RUN: opt -passes="default<O1>" %s -S | FileCheck %s
; REQUIRES: asserts

declare void @bar()
declare void @baz(i32*)

; CHECK-LABEL: @foo1()
define void @foo1() {
entry:
  %tag = alloca i32, align 4
  call void @baz(i32* %tag)
  %tmp = load i32, i32* %tag, align 4
  switch i32 %tmp, label %sw.bb799 [
    i32 10, label %sw.bb239
  ]

sw.bb239:
  call void @foo2()
  br label %cleanup871

sw.bb799:
  call void @foo3(i32 undef)
  br label %cleanup871

cleanup871:
  call void @bar()
  unreachable
}

define void @foo2() {
  call void @foo4()
  unreachable
}

define void @foo3(i32 %ptr) {
  call void @foo1()
  unreachable
}

define void @foo4() {
entry:
  %tag = alloca i32, align 4
  call void @baz(i32* %tag)
  %tmp = load i32, i32* %tag, align 4
  switch i32 %tmp, label %sw.bb442 [
    i32 16, label %sw.bb352
  ]

sw.bb352:
  call void @foo3(i32 undef)
  unreachable

sw.bb442:
  call void @foo2()
  unreachable
}

