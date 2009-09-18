; RUN: llvm-as < %s | llc -march=x86-64 | FileCheck %s

; x86's 32-bit cmov doesn't clobber the high 32 bits of the destination
; if the condition is false. An explicit zero-extend (movl) is needed
; after the cmov.

; CHECK:      cmovne  %edi, %esi
; CHECK-NEXT: movl    %esi, %edi

declare void @bar(i64) nounwind

define void @foo(i64 %a, i64 %b, i1 %p) nounwind {
  %c = trunc i64 %a to i32
  %d = trunc i64 %b to i32
  %e = select i1 %p, i32 %c, i32 %d
  %f = zext i32 %e to i64
  call void @bar(i64 %f)
  ret void
}
