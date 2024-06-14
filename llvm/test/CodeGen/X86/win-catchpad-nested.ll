; RUN: llc -mtriple=x86_64-pc-windows-coreclr < %s | FileCheck %s

declare void @ProcessCLRException()

declare void @f()

define void @test1() personality ptr @ProcessCLRException {
entry:
  invoke void @f()
          to label %exit unwind label %catch.dispatch.1
exit:
  ret void

catch.dispatch.1:
  %cs1 = catchswitch within none [label %outer.catch] unwind to caller

outer.catch:
  %cp1 = catchpad within %cs1 [i32 1]
  invoke void @f() [ "funclet"(token %cp1) ]
          to label %outer.ret unwind label %catch.dispatch.2
outer.ret:
  catchret from %cp1 to label %exit

catch.dispatch.2:
  %cs2 = catchswitch within %cp1 [label %inner.catch] unwind to caller
inner.catch:
  %cp2 = catchpad within %cs2 [i32 2]
  catchret from %cp2 to label %outer.ret
}

; Check the catchret targets
; CHECK-LABEL: test1: # @test1
; CHECK: [[Exit:^[^: ]+]]: # Block address taken
; CHECK-NEXT:              # %exit
; CHECK-NEXT: $ehgcr_0_1:
; CHECK: [[OuterRet:^[^: ]+]]: # Block address taken
; CHECK-NEXT:                  # %outer.ret
; CHECK-NEXT: $ehgcr_0_3:
; CHECK-NEXT: leaq [[Exit]](%rip), %rax
; CHECK:      retq   # CATCHRET
; CHECK: {{^[^: ]+}}: # %inner.catch
; CHECK: .seh_endprolog
; CHECK-NEXT: leaq [[OuterRet]](%rip), %rax
; CHECK:      retq   # CATCHRET
