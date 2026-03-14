; RUN: llvm-link %s %S/Inputs/DSOLocalEquivalent.ll -S | FileCheck %s
; RUN: llvm-link %S/Inputs/DSOLocalEquivalent.ll %s -S | FileCheck %s

declare void @extern_func()
declare void @defined_extern_func()
declare hidden void @hidden_func()

; CHECK:      define void @call_extern_func() {
; CHECK-NEXT:   call void dso_local_equivalent @extern_func()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @call_extern_func() {

  call void dso_local_equivalent @extern_func()
  ret void
}

; CHECK:      define void @call_defined_extern_func() {
; CHECK-NEXT:   call void dso_local_equivalent @defined_extern_func()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @call_defined_extern_func() {
  call void dso_local_equivalent @defined_extern_func()
  ret void
}

; CHECK:      define void @call_hidden_func() {
; CHECK-NEXT:   call void dso_local_equivalent @hidden_func()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @call_hidden_func() {
  call void dso_local_equivalent @hidden_func()
  ret void
}
