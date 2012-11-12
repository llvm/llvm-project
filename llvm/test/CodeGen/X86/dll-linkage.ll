; RUN: llc < %s -mtriple=i386-pc-mingw32 | FileCheck %s

; RUN: llc < %s -mtriple=i386-pc-mingw32 -O0 | FileCheck %s -check-prefix=FAST
; PR6275

declare dllimport void @foo()

define void @bar() nounwind {
; CHECK: calll	*__imp__foo
; FAST:  movl   __imp__foo, [[R:%[a-z]{3}]]
; FAST:  calll  *[[R]]
  call void @foo()
  ret void
}
