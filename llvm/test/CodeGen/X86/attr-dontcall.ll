; RUN: not llc -mtriple=x86_64 -global-isel=0 -fast-isel=0 -stop-after=finalize-isel < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64 -global-isel=0 -fast-isel=1 -stop-after=finalize-isel < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64 -global-isel=1 -fast-isel=0 -stop-after=irtranslator -global-isel-abort=0 < %s 2>&1 | FileCheck %s

declare void @foo() "dontcall"
define void @bar() {
  call void @foo()
  ret void
}

; CHECK: error: call to foo marked "dontcall"
