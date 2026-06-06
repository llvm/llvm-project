; RUN: not llc -mtriple=wasm32-unknown-unknown -filetype=asm %s -o - 2>&1 | FileCheck %s

; CHECK-NOT: LLVM ERROR
; CHECK: in function use i32 (): wasm_var address space symbol must resolve to a GlobalVariable

target triple = "wasm32-unknown-unknown"

@f_alias = alias i32 (), ptr addrspace(1) @f

define i32 @f() addrspace(1) {
  ret i32 0
}

define i32 @use() {
  %v = load i32, ptr addrspace(1) @f_alias
  ret i32 %v
}