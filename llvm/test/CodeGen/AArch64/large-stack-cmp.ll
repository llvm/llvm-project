; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s

define void @foo() {
; CHECK-LABEL: foo:
; CHECK: adds [[TMP:x[0-9]+]], sp,
; CHECK: cmn [[TMP]],

%var = alloca i32, i32 12
  %var2 = alloca i32, i32 1030
  %tst = icmp eq i32* %var, null
  br i1 %tst, label %true, label %false

true:
  call void @bar()
  ret void

false:
  call void @baz()
  ret void
}

declare void @bar()
declare void @baz()
