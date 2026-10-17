; RUN: not llc -mtriple=wasm32-unknown-unknown -filetype=asm %s -o %t 2>&1 | FileCheck %s

; CHECK-NOT: LLVM ERROR
; CHECK: in function foo i32 (): wasm_var address space symbol must resolve to a GlobalVariable

target triple = "wasm32-unknown-unknown"

@bad = alias i32, inttoptr(i32 42 to ptr addrspace(1))

define i32 @foo() {
  %v = load i32, ptr addrspace(1) @bad
  ret i32 %v
}
