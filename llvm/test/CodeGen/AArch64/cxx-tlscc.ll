; RUN: llc < %s -mtriple=aarch64-apple-ios | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-apple-ios -enable-shrink-wrap=true | FileCheck --check-prefix=CHECK %s
; Shrink wrapping currently does not kick in because we have a TLS CALL
; in the entry block and it will clobber the link register.

%struct.S = type { i8 }

@sg = internal thread_local global %struct.S zeroinitializer, align 1
@__dso_handle = external global i8
@__tls_guard = internal thread_local unnamed_addr global i1 false

declare %struct.S* @_ZN1SC1Ev(%struct.S* returned)
declare %struct.S* @_ZN1SD1Ev(%struct.S* returned)
declare i32 @_tlv_atexit(void (i8*)*, i8*, i8*)

define cxx_fast_tlscc nonnull %struct.S* @_ZTW2sg() {
  %.b.i = load i1, i1* @__tls_guard, align 1
  br i1 %.b.i, label %__tls_init.exit, label %init.i

init.i:
  store i1 true, i1* @__tls_guard, align 1
  %call.i.i = tail call %struct.S* @_ZN1SC1Ev(%struct.S* nonnull @sg)
  %1 = tail call i32 @_tlv_atexit(void (i8*)* nonnull bitcast (%struct.S* (%struct.S*)* @_ZN1SD1Ev to void (i8*)*), i8* nonnull getelementptr inbounds (%struct.S, %struct.S* @sg, i64 0, i32 0), i8* nonnull @__dso_handle)
  br label %__tls_init.exit

__tls_init.exit:
  ret %struct.S* @sg
}

; CHECK-LABEL: _ZTW2sg
; CHECK-DAG: stp d31, d30
; CHECK-DAG: stp d29, d28
; CHECK-DAG: stp d27, d26
; CHECK-DAG: stp d25, d24
; CHECK-DAG: stp d23, d22
; CHECK-DAG: stp d21, d20
; CHECK-DAG: stp d19, d18
; CHECK-DAG: stp d17, d16
; CHECK-DAG: stp d7, d6
; CHECK-DAG: stp d5, d4
; CHECK-DAG: stp d3, d2
; CHECK-DAG: stp d1, d0
; CHECK-DAG: stp x20, x19
; CHECK-DAG: stp x14, x13
; CHECK-DAG: stp x12, x11
; CHECK-DAG: stp x10, x9
; CHECK-DAG: stp x8, x7
; CHECK-DAG: stp x6, x5
; CHECK-DAG: stp x4, x3
; CHECK-DAG: stp x2, x1
; CHECK-DAG: stp x29, x30
; CHECK: blr
; CHECK: tbnz w{{.*}}, #0, [[BB_end:.?LBB0_[0-9]+]]
; CHECK: blr
; CHECK: tlv_atexit
; CHECK: [[BB_end]]:
; CHECK: blr
; CHECK-DAG: ldp x2, x1
; CHECK-DAG: ldp x4, x3
; CHECK-DAG: ldp x6, x5
; CHECK-DAG: ldp x8, x7
; CHECK-DAG: ldp x10, x9
; CHECK-DAG: ldp x12, x11
; CHECK-DAG: ldp x14, x13
; CHECK-DAG: ldp x20, x19
; CHECK-DAG: ldp d1, d0
; CHECK-DAG: ldp d3, d2
; CHECK-DAG: ldp d5, d4
; CHECK-DAG: ldp d7, d6
; CHECK-DAG: ldp d17, d16
; CHECK-DAG: ldp d19, d18
; CHECK-DAG: ldp d21, d20
; CHECK-DAG: ldp d23, d22
; CHECK-DAG: ldp d25, d24
; CHECK-DAG: ldp d27, d26
; CHECK-DAG: ldp d29, d28
; CHECK-DAG: ldp d31, d30
