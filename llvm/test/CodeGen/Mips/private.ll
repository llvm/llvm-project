; Test to make sure that the 'private' is used correctly.
;
; RUN: llc -march=mips -relocation-model=pic < %s | FileCheck %s

define private void @foo() {
; CHECK-LABEL: foo:
  ret void
}

@baz = private global i32 4

define i32 @bar() {
; CHECK-LABEL: bar:
; CHECK: call16($foo)
; CHECK: lw $[[R0:[0-9]+]], %got($baz)($
; CHECK: lw ${{[0-9]+}}, %lo($baz)($[[R0]])
  call void @foo()
  %1 = load i32, ptr @baz, align 4
  ret i32 %1
}
