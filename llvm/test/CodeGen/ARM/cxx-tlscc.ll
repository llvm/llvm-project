; RUN: llc < %s -mtriple=armv7k-apple-watchos2.0 | FileCheck %s
; RUN: llc < %s -mtriple=armv7k-apple-watchos2.0 -enable-shrink-wrap=true | FileCheck --check-prefix=CHECK %s
; RUN: llc < %s -mtriple=armv7-apple-ios8.0 | FileCheck %s
; RUN: llc < %s -mtriple=armv7-apple-ios8.0 -enable-shrink-wrap=true | FileCheck --check-prefix=CHECK %s

%struct.S = type { i8 }

@sg = internal thread_local global %struct.S zeroinitializer, align 1
@__dso_handle = external global i8
@__tls_guard = internal thread_local unnamed_addr global i1 false
@sum1 = internal thread_local global i32 0, align 4

declare %struct.S* @_ZN1SC1Ev(%struct.S* returned)
declare %struct.S* @_ZN1SD1Ev(%struct.S* returned)
declare i32 @_tlv_atexit(void (i8*)*, i8*, i8*)

define cxx_fast_tlscc nonnull %struct.S* @_ZTW2sg() nounwind {
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
; CHECK: push {lr}
; CHECK-NOT: push {r1, r2, r3, r4, r7, lr}
; CHECK-NOT: push {r9, r12}
; CHECK-NOT: vpush {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-NOT: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK: blx
; CHECK: bne [[BB_end:.?LBB0_[0-9]+]]
; CHECK; blx
; CHECK: tlv_atexit
; CHECK: [[BB_end]]:
; CHECK: blx
; CHECK-NOT: vpop {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-NOT: vpop {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-NOT: pop {r9, r12}
; CHECK-NOT: pop {r1, r2, r3, r4, r7, pc}
; CHECK: pop {lr}

; CHECK-LABEL: _ZTW4sum1
; CHECK-NOT: push {r1, r2, r3, r4, r7, lr}
; CHECK-NOT: push {r9, r12}
; CHECK-NOT: vpush {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-NOT: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK: blx
define cxx_fast_tlscc nonnull i32* @_ZTW4sum1() nounwind {
  ret i32* @sum1
}
