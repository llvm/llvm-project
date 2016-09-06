; RUN: llc -verify-machineinstrs < %s --enable-shrink-wrap=false -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s
%struct.S = type { i8 }

@sg = internal thread_local global %struct.S zeroinitializer, align 1
@__dso_handle = external global i8
@__tls_guard = internal thread_local unnamed_addr global i1 false
@sum1 = internal thread_local global i32 0, align 4

declare void @_ZN1SC1Ev(%struct.S*)
declare void @_ZN1SD1Ev(%struct.S*)
declare i32 @_tlv_atexit(void (i8*)*, i8*, i8*)

; CHECK-LABEL: _ZTW2sg
define cxx_fast_tlscc nonnull %struct.S* @_ZTW2sg() nounwind {
  %.b.i = load i1, i1* @__tls_guard, align 1
; CHECK: bc 12, 1, [[BB_end:.?LBB0_[0-9]+]]
  br i1 %.b.i, label %__tls_init.exit, label %init.i

init.i:
; CHECK: Folded Spill
  store i1 true, i1* @__tls_guard, align 1
  tail call void @_ZN1SC1Ev(%struct.S* nonnull @sg) #2
; CHECK: bl _ZN1SC1Ev
  %1 = tail call i32 @_tlv_atexit(void (i8*)* nonnull bitcast (void (%struct.S*)* @_ZN1SD1Ev to void (i8*)*), i8* nonnull getelementptr inbounds (%struct.S, %struct.S* @sg, i64 0, i32 0), i8* nonnull @__dso_handle) #2
; CHECK: Folded Reload
; CHECK: _tlv_atexit
  br label %__tls_init.exit

; CHECK: [[BB_end]]:
__tls_init.exit:
  ret %struct.S* @sg
}

; CHECK-LABEL: _ZTW4sum1
define cxx_fast_tlscc nonnull i32* @_ZTW4sum1() nounwind {
  ret i32* @sum1
}

define cxx_fast_tlscc i32* @_ZTW4sum2() #0 {
  ret i32* @sum1
}

attributes #0 = { nounwind "no-frame-pointer-elim"="true" }