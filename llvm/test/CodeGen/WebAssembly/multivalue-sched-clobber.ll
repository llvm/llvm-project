; RUN: llc < %s -verify-machineinstrs -mattr=+multivalue -target-abi=experimental-mv -pre-RA-sched=list-burr | FileCheck %s

; The results of a multivalue call are virtual registers, not implicit
; physical register defs. The list scheduler used to index the implicit def
; list of CALL_RESULTS by result number for them and assert in
; canClobberPhysRegDefs when the second result was used.

target triple = "wasm32-unknown-unknown"

declare { i32, i32 } @g(i32)
declare void @h(i32)

; CHECK-LABEL: f:
; CHECK: call g
; CHECK: call h
; CHECK: end_function
define i32 @f(i32 %p) {
  %s = call { i32, i32 } @g(i32 %p)
  %r = extractvalue { i32, i32 } %s, 1
  call void @h(i32 %p)
  ret i32 %r
}
