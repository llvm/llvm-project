; RUN: llc < %s -mtriple=i386-apple-darwin9   | FileCheck %s -check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s -check-prefix=X64

%struct.p = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

define i32 @f(ptr byval(%struct.p) align 4 %q) nounwind ssp {
entry:
; X86: _f:
; X86: jmp _g

; X64: _f:
; X64: jmp _g
  %call = tail call i32 @g(ptr byval(%struct.p) align 4 %q) nounwind
  ret i32 %call
}

declare i32 @g(ptr byval(%struct.p) align 4)

define i32 @h(ptr byval(%struct.p) align 4 %q, i32 %r) nounwind ssp {
entry:
; X86: _h:
; X86: jmp _i

; X64: _h:
; X64: jmp _i

  %call = tail call i32 @i(ptr byval(%struct.p) align 4 %q, i32 %r) nounwind
  ret i32 %call
}

declare i32 @i(ptr byval(%struct.p) align 4, i32)
