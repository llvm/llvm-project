; RUN: not llc -mtriple=x86_64 -global-isel=0 -fast-isel=0 -stop-after=finalize-isel < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64 -global-isel=0 -fast-isel=1 -stop-after=finalize-isel < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64 -global-isel=1 -fast-isel=0 -stop-after=irtranslator -global-isel-abort=0 < %s 2>&1 | FileCheck %s

declare void @foo() "dontcall-error"="e"
define void @bar() {
  call void @foo()
  ret void
}

declare void @foo2() "dontcall-warn"="w"
define void @bar2() {
  call void @foo2()
  ret void
}

declare void @foo3() "dontcall-warn"
define void @bar3() {
  call void @foo3()
  ret void
}

; CHECK: error: call to foo marked "dontcall-error": e
; CHECK: warning: call to foo2 marked "dontcall-warn": w
; CHECK: warning: call to foo3 marked "dontcall-warn"{{$}}
