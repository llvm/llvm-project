; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false %s -o - | FileCheck --check-prefixes CHECK -DPTR=i32 %s
; RUN: llc --mtriple=wasm64-unknown-unknown -asm-verbose=false %s -o - | FileCheck --check-prefixes CHECK -DPTR=i64 %s

; empty function that does not use stack, to check WebAssemblyMCLowerPrePass
; works correctly.
define hidden void @bar() #0 {
  ret void
}

; Function that uses explict stack, and should generate a reference to
; __stack_pointer, along with the corresponding relocation entry.
define hidden void @foo() #0 {
entry:
  alloca i32, align 4
  ret void
}

; CHECK:              .text
; CHECK-NEXT:         .file   "stack-ptr-mclower.ll"
; CHECK-NEXT:         .globaltype     __stack_pointer, [[PTR]]
; CHECK-NEXT:         .section        .text.bar,"",@
; CHECK-NEXT:         .hidden bar
; CHECK-NEXT:         .globl  bar
; CHECK-NEXT:         .type   bar,@function
; CHECK-NEXT: bar:
; CHECK-NEXT:         .functype       bar () -> ()
; CHECK-NEXT:         end_function
; CHECK-NEXT: .Lfunc_end0:
; CHECK-NEXT:         .size   bar, .Lfunc_end0-bar

; CHECK:              .section        .text.foo,"",@
; CHECK-NEXT:         .hidden foo
; CHECK-NEXT:         .globl  foo
; CHECK-NEXT:         .type   foo,@function
; CHECK-NEXT: foo:
; CHECK-NEXT:         .functype       foo () -> ()
; CHECK-NEXT:         global.get      __stack_pointer
; CHECK-NEXT:         [[PTR]].const       16
; CHECK-NEXT:         [[PTR]].sub
; CHECK-NEXT:         drop
; CHECK-NEXT:         end_function
; CHECK-NEXT: .Lfunc_end1:
; CHECK-NEXT:         .size   foo, .Lfunc_end1-foo
