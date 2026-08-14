; RUN: %python %S/Inputs/generate-di-expression-symbolic-branch-out-of-range.py > %t.ll
; RUN: not --crash llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o /dev/null %t.ll 2>&1 | FileCheck %s

; The first positive offset that doesn't fit is 32768, so make sure CodeGen
; reports it instead of truncating it.

; CHECK: LLVM ERROR: DWARF expression branch offset 32768 is outside [-32768, 32767]
