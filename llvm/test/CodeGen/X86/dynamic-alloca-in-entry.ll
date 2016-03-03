; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

; Allocas with unknown size in the entry block are dynamic.
define void @foo(i32 %n) {
  %m = alloca i32, i32 %n
  ret void
}
; CHECK-LABEL: _foo:
; CHECK: calll __chkstk
; CHECK: retl

; Use of inalloca implies that that the alloca is not static.
define void @bar() {
  %m = alloca inalloca i32
  ret void
}
; CHECK-LABEL: _bar:
; CHECK: calll __chkstk
; CHECK: retl
