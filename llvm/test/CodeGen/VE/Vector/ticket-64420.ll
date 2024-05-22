; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s
; RUN: llc < %s -mtriple=ve -mattr=-vpu | FileCheck --check-prefix=SCALAR %s

; Check vector and scalar code generation for vector load instruction.
; For the case of vector, generates vst with 4 vector length.  For the
; case of scalar, generates 2 stores of 8 bytes length.

; This is taken from a ticket below.
;   https://github.com/llvm/llvm-project/issues/64420

; CHECK-LABEL: func:
; CHECK:       # %bb.1:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, 0
; CHECK-NEXT:    or %s1, 4, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vstl %v0, 4, %s0
; CHECK-NEXT:    b.l.t (, %s10)

; SCALAR-LABEL: func:
; SCALAR:       # %bb.1:
; SCALAR-NEXT:    st %s1, 8(, %s0)
; SCALAR-NEXT:    st %s1, (, %s0)
; SCALAR-NEXT:    b.l.t (, %s10)

; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "test.c"
target datalayout = "e-m:e-i64:64-n32:64-S128-v64:64:64-v128:64:64-v256:64:64-v512:64:64-v1024:64:64-v2048:64:64-v4096:64:64-v8192:64:64-v16384:64:64"
target triple = "ve-unknown-linux-gnu"

define dso_local void @func(ptr %_0) unnamed_addr #0 {
start:
  br i1 poison, label %bb7, label %panic3

bb7:                                              ; preds = %start
  store <4 x i32> zeroinitializer, ptr %_0, align 4
  ret void

panic3:                                           ; preds = %start
  unreachable
}

attributes #0 = { "target-features"="+vpu" }
