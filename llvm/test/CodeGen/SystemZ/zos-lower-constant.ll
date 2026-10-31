; Test lowering of constants on z/OS
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; A pointer to a global variable is emitted as a direct AD-con.
; CHECK-LABEL: ptr_var CSECT
; CHECK:  DC AD(gvar)

; A pointer to an internal function is emitted as an offset into the ADA
; (function descriptor of foo is the first ADA slot, at offset 0).
; CHECK-LABEL: ptr_foo CSECT
; CHECK:  DC AD(AD({{.*}}#S)+XL8'0000000000000000')

; A pointer to a function alias with external linkage uses the indirect symbol.
; CHECK-LABEL: ptr_bar CSECT
; CHECK:  DC VD(bar@indirect)

; The ADA section holds: the function descriptor of foo (internal, offset 0)
; and the indirect slot for bar (external alias, offset 16).
; CHECK-LABEL: {{.*}}#S XATTR
; CHECK: * Offset 0 function descriptor of foo
; CHECK-NEXT:  DC RD(foo)
; CHECK-NEXT:  DC VD(foo)
; CHECK: * Offset 16 pointer to function descriptor bar
; CHECK-NEXT:  DC VD(bar@indirect)

@gvar = hidden global i32 42, align 4

; GV path in lowerConstant: pointer to a global variable.
@ptr_var = hidden global ptr @gvar, align 8

define internal void @foo() {
entry:
  ret void
}

; FV path in lowerConstant: pointer to an internal function (descriptor in ADA).
@ptr_foo = hidden global ptr @foo, align 8

; GA path in lowerConstant: alias of a function with external linkage.
@bar = hidden alias void (), ptr @foo

; GA path with non-local linkage: uses the indirect symbol.
@ptr_bar = hidden global ptr @bar, align 8
