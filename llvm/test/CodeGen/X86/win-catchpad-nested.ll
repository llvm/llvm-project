; RUN: llc -mtriple=x86_64-pc-windows-coreclr < %s | FileCheck %s

declare void @ProcessCLRException()

declare void @f()

define void @test1() personality void ()* @ProcessCLRException {
entry:
  invoke void @f()
          to label %exit unwind label %outer.pad
outer.pad:
  %outer = catchpad [i32 1]
          to label %outer.catch unwind label %outer.end
outer.catch:
  invoke void @f()
          to label %outer.ret unwind label %inner.pad
inner.pad:
  %inner = catchpad [i32 2]
          to label %inner.ret unwind label %inner.end
inner.ret:
  catchret %inner to label %outer.ret
inner.end:
  catchendpad unwind label %outer.end
outer.ret:
  catchret %outer to label %exit
outer.end:
  catchendpad unwind to caller
exit:
  ret void
}

; Check the catchret targets
; CHECK-LABEL: test1: # @test1
; CHECK: [[Exit:^[^: ]+]]: # Block address taken
; CHECK-NEXT:              # %exit
; CHECK: [[OuterRet:^[^: ]+]]: # Block address taken
; CHECK-NEXT:                  # %outer.ret
; CHECK-NEXT: leaq [[Exit]](%rip), %rax
; CHECK:      retq   # CATCHRET
; CHECK: {{^[^: ]+}}: # %inner.pad
; CHECK: .seh_endprolog
; CHECK-NEXT: leaq [[OuterRet]](%rip), %rax
; CHECK:      retq   # CATCHRET
