; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f(token %A, token %B) {
entry:
  br label %bb

bb:
  %sel = select i1 undef, token %A, token %B
; CHECK: select values cannot have token type
  br label %bb
}
