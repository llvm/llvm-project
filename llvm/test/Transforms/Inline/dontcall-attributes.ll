; RUN: opt -S -passes=inline < %s | FileCheck %s --check-prefixes=CHECK-BOTH,CHECK
; RUN: opt -S -passes=always-inline < %s | FileCheck %s --check-prefixes=CHECK-BOTH,CHECK-ALWAYS

declare void @foo() "dontcall-warn"="oh no"
declare void @fof() "dontcall-error"="oh no"

define void @bar(i32 %x) {
  %cmp = icmp eq i32 %x, 10
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @foo()
  br label %if.end

if.end:
  ret void
}

define void @quux() {
  call void @bar(i32 9)
  ret void
}

; Test that @baz's call to @foo has metadata with inlining info.
define void @baz() {
; CHECK-LABEL: define {{[^@]+}}@baz(
; CHECK-NEXT:    call void @foo(), !inlined.from !0
  call void @bar(i32 10)
  ret void
}

; Test that @zing's call to @foo has unique metadata from @baz's call to @foo.
define void @zing() {
; CHECK-LABEL: define {{[^@]+}}@zing(
; CHECK-NEXT:    call void @foo(), !inlined.from !1
  call void @baz()
  ret void
}

; Same test but @fof has fn attr "dontcall-error"="..." rather than
; "dontcall-warn"="...".
define void @_Z1av() {
  call void @fof()
  ret void
}
define void @_Z1bv() {
; CHECK-LABEL: define {{[^@]+}}@_Z1bv(
; CHECK-NEXT: call void @fof(), !inlined.from !3
  call void @_Z1av()
  ret void
}

; Add some tests for alwaysinline.
define void @always_callee() alwaysinline {
  call void @fof()
  ret void
}
define void @always_caller() alwaysinline {
; CHECK-BOTH-LABEL: define {{[^@]+}}@always_caller(
; CHECK-NEXT: call void @fof(), !inlined.from !4
; CHECK-ALWAYS-NEXT: call void @fof(), !inlined.from !0
  call void @always_callee()
  ret void
}
define void @always_caller2() alwaysinline {
; CHECK-BOTH-LABEL: define {{[^@]+}}@always_caller2(
; CHECK-NEXT: call void @fof(), !inlined.from !5
; CHECK-ALWAYS-NEXT: call void @fof(), !inlined.from !1
  call void @always_caller()
  ret void
}

; CHECK: !0 = !{!"bar"}
; CHECK-NEXT: !1 = !{!2}
; CHECK-NEXT: !2 = !{!"bar", !"baz"}
; CHECK-NEXT: !3 = !{!"_Z1av"}
; CHECK-NEXT: !4 = !{!"always_callee"}
; CHECK-ALWAYS: !0 = !{!"always_callee"}
; CHECK-ALWAYS-NEXT: !1 = !{!2}
; CHECK-ALWAYS-NEXT: !2 = !{!"always_callee", !"always_caller"}
