; RUN: split-file %s %t
; RUN: llvm-diff %t/a.ll %t/a.ll
; RUN: not llvm-diff %t/a.ll %t/b.ll 2>&1 | FileCheck %s --implicit-check-not=same

; CHECK-LABEL: in function decl:
; CHECK-NEXT: function attributes differ:
; CHECK-NEXT: < nounwind
; CHECK-NEXT: > noreturn
; CHECK-LABEL: in function function_attrs:
; CHECK-NEXT: function attributes differ:
; CHECK-NEXT: > noinline
; CHECK-LABEL: in function return_attrs:
; CHECK-NEXT: return attributes differ:
; CHECK-NEXT: < noundef
; CHECK-LABEL: in function parameter_attrs:
; CHECK-NEXT: parameter 0 attributes differ:
; CHECK-NEXT: < noundef
; CHECK-NEXT: parameter 1 attributes differ:
; CHECK-NEXT: > noundef
; CHECK-LABEL: in function calling_conv:
; CHECK-NEXT: different calling conventions
; CHECK-LABEL: in function call_attrs:
; CHECK: {{^ *>}} {{.*}}call void @callee(i32 %x){{$}}
; CHECK-NEXT: {{^ *<}} {{.*}}call void @callee(i32 %x) #{{[0-9]+}}
; CHECK-LABEL: in function call_parameter_attrs:
; CHECK: {{^ *>}} {{.*}}call void @callee(i32 %x){{$}}
; CHECK-NEXT: {{^ *<}} {{.*}}call void @callee(i32 noundef %x)

;--- a.ll
declare void @callee(i32)
declare void @decl() nounwind

define void @function_attrs() nounwind {
  ret void
}

define noundef i32 @return_attrs(i32 %x) {
  ret i32 %x
}

define void @parameter_attrs(i32 noundef %x, i32 %y) {
  ret void
}

define void @calling_conv() {
  ret void
}

define void @call_attrs(i32 %x) {
  call void @callee(i32 %x) nounwind
  ret void
}

define void @call_parameter_attrs(i32 %x) {
  call void @callee(i32 noundef %x)
  ret void
}

define void @same(i32 noundef %x) nounwind {
  call void @callee(i32 %x) nounwind
  ret void
}

;--- b.ll
declare void @callee(i32)
declare void @decl() noreturn

define void @function_attrs() noinline nounwind {
  ret void
}

define i32 @return_attrs(i32 %x) {
  ret i32 %x
}

define void @parameter_attrs(i32 %x, i32 noundef %y) {
  ret void
}

define fastcc void @calling_conv() {
  ret void
}

define void @call_attrs(i32 %x) {
  call void @callee(i32 %x)
  ret void
}

define void @call_parameter_attrs(i32 %x) {
  call void @callee(i32 %x)
  ret void
}

define void @same(i32 noundef %x) nounwind {
  call void @callee(i32 %x) nounwind
  ret void
}
