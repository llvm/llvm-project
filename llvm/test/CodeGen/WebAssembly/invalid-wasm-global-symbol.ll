; RUN: not --crash llc -mtriple=wasm32-unknown-unknown -filetype=asm %s -o - 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: wasm_var address space symbol must resolve to a GlobalVariable

target triple = "wasm32-unknown-unknown"

@bad = alias i32, inttoptr(i32 42 to ptr addrspace(1))

define i32 @foo() {
  %v = load i32, ptr addrspace(1) @bad
  ret i32 %v
}