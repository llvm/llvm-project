; RUN: not llc -mtriple=x86_64 -global-isel=0 -fast-isel=0 -stop-after=finalize-isel < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64 -global-isel=0 -fast-isel=1 -stop-after=finalize-isel < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64 -global-isel=1 -fast-isel=0 -global-isel-abort=0 < %s 2>&1 | FileCheck %s

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

declare void @foo4(i32) addrspace(1) "dontcall-warn"="cast"

define void @bar4() {
  call void addrspacecast (ptr addrspace(1) @foo4 to ptr)(i32 0)
  ret void
}

declare i32 @_Z3fooi(i32) "dontcall-error"
define void @demangle1() {
  call i32 @_Z3fooi (i32 0)
  ret void
}
declare float @_Z3barf(float) "dontcall-error"
define void @demangle2() {
  call float @_Z3barf(float 0.0)
  ret void
}

declare i32 @_RNvC1a3baz() "dontcall-error"
define void @demangle3() {
  call i32 @_RNvC1a3baz()
  ret void
}


declare i32 @_Z3fooILi79EEbU7_ExtIntIXT_EEi(i32) "dontcall-error"
define void @demangle4() {
  call i32 @_Z3fooILi79EEbU7_ExtIntIXT_EEi(i32 0)
  ret void
}
; CHECK: error: call to foo marked "dontcall-error": e
; CHECK: warning: call to foo2 marked "dontcall-warn": w
; CHECK: warning: call to foo3 marked "dontcall-warn"{{$}}
; CHECK: warning: call to foo4 marked "dontcall-warn": cast
; CHECK: error: call to foo(int) marked "dontcall-error"
; CHECK: error: call to bar(float) marked "dontcall-error"
; CHECK: error: call to a::baz marked "dontcall-error"
; CHECK: error: call to bool foo<79>(int _ExtInt<79>) marked "dontcall-error"
