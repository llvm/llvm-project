; RUN: llc -fast-isel -O0 -code-model=large -mcpu=generic -mtriple=x86_64-apple-darwin10 -relocation-model=pic < %s | FileCheck %s

; Check that fast-isel cleans up when it fails to lower a call instruction.
define void @fastiselcall() {
entry:
  %call = call i32 @targetfn(i32 42)
  ret void
; CHECK-LABEL: fastiselcall:
; Local value area is still there:
; CHECK: movl $42, {{%[a-z]+}}
; Fast-ISel's arg mov is not here:
; CHECK-NOT: movl $42, (%esp)
; SDag-ISel's arg mov:
; CHECK: movabsq $_targetfn, %[[REG:[^ ]*]]
; CHECK: movl $42, %edi
; CHECK: callq *%[[REG]]

}
declare i32 @targetfn(i32)
