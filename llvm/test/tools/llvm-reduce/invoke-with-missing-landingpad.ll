; RUN: llvm-reduce --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s  -abort-on-invalid-reduction -o %t
; RUN: FileCheck <%t %s

; CHECK-INTERESTINGNESS: call void @foo()

; CHECK: define void @test() personality ptr null {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %cont
; CHECK-EMPTY:
; CHECK-NEXT: cont:
; CHECK-NEXT:   br label %exit
; CHECK-EMPTY:
; CHECK-NEXT: exit:
; CHECK-NEXT:   call void @foo()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @test() personality ptr null {
entry:
  invoke void @foo()
          to label %cont unwind label %lpad

cont:
  invoke void @foo()
          to label %exit unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
  ret void

exit:
  call void @foo()
  ret void
}

declare void @foo()
