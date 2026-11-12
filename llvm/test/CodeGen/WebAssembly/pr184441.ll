; RUN: llc < %s -O0 -verify-machineinstrs
target triple = "wasm32-unknown-unknown"

; Regression test for #184441.

; Ensures that the orphanation of block %b due to the eliding of the
; `unreachable` default label of the switch during IR => DAG translation
; (without any subsequent DCE) doesn't trip our assertions in
; WebAssemblyFixIrreducibleControlFlow, which expect all blocks in a function
; to be reachable from the function's entry.

define void @test() {
entry:
  switch i32 poison, label %b [
    i32 0, label %a
  ]
a:
  ret void
b:
  unreachable
}
