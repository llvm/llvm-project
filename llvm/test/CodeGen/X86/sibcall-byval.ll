; RUN: llc < %s -mtriple=i386-apple-darwin   | FileCheck %s -check-prefix=32
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s -check-prefix=64

%struct.p = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

define i32 @f(%struct.p* byval align 4 %q) nounwind ssp {
entry:
; 32: _f:
; 32: jmp L_g$stub

; 64: _f:
; 64: jmp _g
  %call = tail call i32 @g(%struct.p* byval align 4 %q) nounwind
  ret i32 %call
}

declare i32 @g(%struct.p* byval align 4)

define i32 @h(%struct.p* byval align 4 %q, i32 %r) nounwind ssp {
entry:
; 32: _h:
; 32: jmp L_i$stub

; 64: _h:
; 64: jmp _i

  %call = tail call i32 @i(%struct.p* byval align 4 %q, i32 %r) nounwind
  ret i32 %call
}

declare i32 @i(%struct.p* byval align 4, i32)
