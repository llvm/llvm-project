; Test lowering of constants on z/OS
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; A pointer to a function that is not internal refers to the function
; descriptor through the indirect symbol, like code taking the address
; (@take_bar), so that both compare equal. Both use the same ADA slot.
; CHECK: func_s CSECT
; CHECK: DC AD(AD({{.*}}#S)+XL8'8')
; CHECK: func_e CSECT
; CHECK: DC VD(bar@indirect)
; CHECK: * Offset 0 pointer to function descriptor bar
; CHECK-NEXT: DC VD(bar@indirect)
; CHECK-NEXT: * Offset 8 function descriptor of foo
; CHECK-NEXT: DC RD(foo)
; CHECK-NEXT: DC VD(foo)
; CHECK-NOT: DC VD(bar@indirect)
@x = hidden global i32 4077, align 4
@y = hidden global ptr @x, align 8
@func_s = hidden global ptr @foo, align 8
@func_e = hidden global ptr @bar, align 8

define hidden void @bar() {
entry:
  ret void
}

define internal void @foo() {
entry:
  ret void
}

define hidden ptr @take_bar() {
entry:
  ret ptr @bar
}
