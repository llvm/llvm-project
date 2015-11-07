; RUN: llc < %s -asm-verbose=false | FileCheck %s
; RUN: llc < %s -asm-verbose=false -fast-isel | FileCheck %s

; Test that wasm select instruction is selected from LLVM select instruction.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: select_i32:
; CHECK: get_local push, 1
; CHECK: set_local [[LOCAL_B:[0-9]]]
; CHECK: get_local push, 0
; CHECK: set_local [[LOCAL_A:[0-9]]]
; CHECK: i32.eq push, (get_local 5), (get_local 6)
; CHECK: set_local 7, pop
; CHECK: i32.select push, (get_local 7), (get_local [[LOCAL_A]]), (get_local [[LOCAL_B]])
define i32 @select_i32(i32 %a, i32 %b, i32 %cond) {
 %cc = icmp eq i32 %cond, 0
 %result = select i1 %cc, i32 %a, i32 %b
 ret i32 %result
}

; CHECK-LABEL: select_i64:
; CHECK: get_local push, 1
; CHECK: set_local [[LOCAL_B:[0-9]]]
; CHECK: get_local push, 0
; CHECK: set_local [[LOCAL_A:[0-9]]]
; CHECK: i32.eq push, (get_local 5), (get_local 6)
; CHECK: set_local 7, pop
; CHECK: i64.select push, (get_local 7), (get_local [[LOCAL_A]]), (get_local [[LOCAL_B]])
define i64 @select_i64(i64 %a, i64 %b, i32 %cond) {
 %cc = icmp eq i32 %cond, 0
 %result = select i1 %cc, i64 %a, i64 %b
 ret i64 %result
}

; CHECK-LABEL: select_f32:
; CHECK: get_local push, 1
; CHECK: set_local [[LOCAL_B:[0-9]]]
; CHECK: get_local push, 0
; CHECK: set_local [[LOCAL_A:[0-9]]]
; CHECK: i32.eq push, (get_local 5), (get_local 6)
; CHECK: set_local 7, pop
; CHECK: f32.select push, (get_local 7), (get_local [[LOCAL_A]]), (get_local [[LOCAL_B]])
define float @select_f32(float %a, float %b, i32 %cond) {
 %cc = icmp eq i32 %cond, 0
 %result = select i1 %cc, float %a, float %b
 ret float %result
}

; CHECK-LABEL: select_f64:
; CHECK: get_local push, 1
; CHECK: set_local [[LOCAL_B:[0-9]]]
; CHECK: get_local push, 0
; CHECK: set_local [[LOCAL_A:[0-9]]]
; CHECK: i32.eq push, (get_local 5), (get_local 6)
; CHECK: set_local 7, pop
; CHECK: f64.select push, (get_local 7), (get_local [[LOCAL_A]]), (get_local [[LOCAL_B]])
define double @select_f64(double %a, double %b, i32 %cond) {
 %cc = icmp eq i32 %cond, 0
 %result = select i1 %cc, double %a, double %b
 ret double %result
}
