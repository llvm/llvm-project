; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; We should treat non-Function personalities as the unknown personality, which
; is usually Itanium.

declare void @g()
declare void @terminate(ptr)

define void @f() personality ptr null {
  invoke void @g()
    to label %ret unwind label %lpad
ret:
  ret void
lpad:
  %vals = landingpad { ptr, i32 } catch ptr null
  %ptr = extractvalue { ptr, i32 } %vals, 0
  call void @terminate(ptr %ptr)
  unreachable
}

; CHECK: f:
; CHECK: callq g
; CHECK: retq
; CHECK: movq %rax, %rdi
; CHECK: callq terminate
