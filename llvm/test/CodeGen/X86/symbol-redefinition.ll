; RUN: not llc -mtriple=x86_64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; CHECK: <unknown>:0: error: symbol 'fn' is already defined
define void @fn() section "fn" {
  ret void
}

; CHECK: <unknown>:0: error: symbol 'var' is already defined
@var = global i32 0, section "var", align 4
