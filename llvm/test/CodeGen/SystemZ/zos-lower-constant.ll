; Test lowering of constants on z/OS
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; CHECK: func_s CSECT
; CHECK: DC AD(AD({{.*}}#S)+XL8'0')
; CHECK: func_e CSECT
; CHECK: DC VD(bar)
; CHECK: DC RD(foo)
; CHECK-NEXT: DC VD(foo)
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
