; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s

; CodeGen should remat the zero instead of spilling it.

declare void @foo(i64 %p)

; CHECK-LABEL: bar:
; CHECK: xorl %e[[A0:di|cx]], %e
; CHECK: xorl %e[[A0]], %e[[A0]]
define void @bar() nounwind {
  call void @foo(i64 0)
  call void @foo(i64 0)
  ret void
}

; CHECK-LABEL: bat:
; CHECK: movq $-1, %r[[A0]]
; CHECK: movq $-1, %r[[A0]]
define void @bat() nounwind {
  call void @foo(i64 -1)
  call void @foo(i64 -1)
  ret void
}

; CHECK-LABEL: bau:
; CHECK: movl $1, %e[[A0]]
; CHECK: movl $1, %e[[A0]]
define void @bau() nounwind {
  call void @foo(i64 1)
  call void @foo(i64 1)
  ret void
}

