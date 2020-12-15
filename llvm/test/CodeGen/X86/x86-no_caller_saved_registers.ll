; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-unknown -O0 < %s | FileCheck %s
; RUN: llc -mtriple=i686-unknown-unknown -mattr=+sse2 < %s | FileCheck %s
; RUN: llc -mtriple=i686-unknown-unknown -mattr=+sse2 -O0 < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; In functions with 'no_caller_saved_registers' attribute, all registers should
;; be preserved except for registers used for passing/returning arguments.
;; The test checks that function "bar" preserves xmm0 register.
;; It also checks that caller function "foo" does not store registers for callee
;; "bar". For example, there is no store/load/access to xmm registers.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i32 @bar(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8) #0 {
; CHECK-LABEL: bar
; CHECK:       mov{{.*}}  %xmm0
; CHECK:       mov{{.*}} {{.*}}, %xmm0
; CHECK:       ret
  call void asm sideeffect "", "~{xmm0}"()
  ret i32 1
}

define x86_intrcc void @foo(i8* byval(i8) nocapture readnone %c) {
; CHECK-LABEL: foo
; CHECK-NOT: xmm
entry:
  tail call i32 @bar(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8) #0
  ret void
}

attributes #0 = { "no_caller_saved_registers" }
