; Test the return of composite values with -freg-struct-return.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define { i64, i32 } @ret_i64_i32(ptr %p) {
; CHECK-LABEL: ret_i64_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lg %r0, 0(%r2)
; CHECK-NEXT:    l %r3, 8(%r2)
; CHECK-NEXT:    lgr %r2, %r0
; CHECK-NEXT:    br %r14
  %v = load { i64, i32 }, ptr %p
  ret { i64, i32 } %v
}

define void @call_i64_i32(ptr %p) {
; CHECK-LABEL: call_i64_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stmg %r13, %r15, 104(%r15)
; CHECK-NEXT:    .cfi_offset %r13, -56
; CHECK-NEXT:    .cfi_offset %r14, -48
; CHECK-NEXT:    .cfi_offset %r15, -40
; CHECK-NEXT:    aghi %r15, -160
; CHECK-NEXT:    .cfi_def_cfa_offset 320
; CHECK-NEXT:    lgr %r13, %r2
; CHECK-NEXT:    brasl %r14, ret_i64_i32@PLT
; CHECK-NEXT:    stg %r2, 0(%r13)
; CHECK-NEXT:    st %r3, 8(%r13)
; CHECK-NEXT:    lmg %r13, %r15, 264(%r15)
; CHECK-NEXT:    br %r14
  %v = call { i64, i32 } @ret_i64_i32(ptr %p)
  store { i64, i32 } %v, ptr %p
  ret void
}
