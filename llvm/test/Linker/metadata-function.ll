; RUN: llvm-link %s %S/Inputs/metadata-function.ll -S | FileCheck %s
; RUN: llvm-link %S/Inputs/metadata-function.ll %s -S | FileCheck %s

; CHECK-DAG: define weak void @foo() !weak ![[B:[0-9]+]] {
define linkonce void @foo() !linkonce !0 {
  unreachable
}

; CHECK-DAG: define void @bar() !bar ![[A:[0-9]+]] {
define void @bar() !bar !0 {
  call void @baz()
  unreachable
}

; CHECK-DAG: define void @baz() !baz ![[B]] {
declare void @baz()

; CHECK-DAG: define void @a() !a ![[A]] {
; CHECK-DAG: define void @b() !b ![[B]] {
define void @a() !a !0 {
  unreachable
}

; CHECK-DAG: define %[[HandleType:[A-Za-z]+]] @init.Handle() {
; CHECK-DAG: declare !types ![[C:[0-9]+]] %[[HandleType]] @init.AltHandle()
; CHECK-DAG: define void @uses.AltHandle() {
%Handle = type { i8* }
define %Handle @init.Handle() {
  unreachable
}

; CHECK-DAG: ![[A]] = !{!"a"}
; CHECK-DAG: ![[B]] = !{!"b"}
; CHECK-DAG: ![[C]] = !{%[[HandleType]] undef}
!0 = !{!"a"}
