; RUN: not llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s --implicit-check-not=error:

.code

; CHECK: :[[# @LINE + 2]]:5: error: Expected @@ label before @B reference
; CHECK: :[[# @LINE + 1]]:7: error: Unexpected identifier!
jmp @B

@@:
  jmp @B
  jmp @F
@@:
  xor eax, eax

; NOTE: a trailing @F will not fail; fixing this seems to require two passes.
jmp @F
