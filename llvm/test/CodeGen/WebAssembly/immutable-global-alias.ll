; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false | FileCheck %s

@defined_g = addrspace(1) constant i32 42
@defined_g_alias = alias i32, ptr addrspace(1) @defined_g

define i32 @foo() {
; CHECK-LABEL: foo:
; CHECK-NEXT:  .functype       foo () -> (i32)
; CHECK-NEXT:  global.get      defined_g_alias
; CHECK-NEXT:  end_function
  %v = load i32, ptr addrspace(1) @defined_g_alias
  ret i32 %v
}

; CHECK:       .globaltype     defined_g, i32, immutable
; CHECK:       defined_g_alias = defined_g