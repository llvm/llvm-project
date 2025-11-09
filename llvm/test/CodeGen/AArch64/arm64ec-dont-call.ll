; RUN: not llc -mtriple=arm64ec-windows-msvc -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=arm64ec-windows-msvc -filetype=null -global-isel=1 -global-isel-abort=0 %s 2>&1 | FileCheck %s

define void @baz() #0 {
  call void @foo()
  ret void
}

define void @foo() #1 {
  ret void
}

attributes #0 = { noinline optnone }
attributes #1 = { "dontcall-error"="oh no foo" }

; Regression test for `dontcall-error` for Arm64EC. Since this attribute is
; checked both by FastISel and SelectionDAGBuilder, and FastISel was bailing for
; Arm64EC AFTER doing the check, we ended up with duplicate copies of this
; error.

; CHECK: error: call to #foo marked "dontcall-error": oh no foo
; CHECK-NOT: error:
