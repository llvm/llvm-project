; RUN: llc < %s -mtriple=armv7k-apple-watchos2.0 | FileCheck %s
; RUN: llc < %s -mtriple=armv7k-apple-watchos2.0 -enable-shrink-wrap=true | FileCheck %s
; RUN: llc < %s -mtriple=armv7-apple-ios8.0 | FileCheck %s
; RUN: llc < %s -mtriple=armv7-apple-ios8.0 -enable-shrink-wrap=true | FileCheck %s

; RUN: llc < %s -mtriple=armv7k-apple-watchos2.0 -O0 | FileCheck --check-prefix=CHECK-O0 --check-prefix=WATCH-O0 %s
; RUN: llc < %s -mtriple=armv7-apple-ios8.0 -O0 | FileCheck --check-prefix=CHECK-O0 --check-prefix=IOS-O0 %s

; RUN: llc < %s -mtriple=thumbv7-apple-ios8.0 | FileCheck --check-prefix=THUMB %s

%struct.S = type { i8 }

@sg = internal thread_local global %struct.S zeroinitializer, align 1
@__dso_handle = external global i8
@__tls_guard = internal thread_local unnamed_addr global i1 false
@sum1 = internal thread_local global i32 0, align 4

%class.C = type { i32 }
@tC = internal thread_local global %class.C zeroinitializer, align 4

declare ptr @_ZN1SC1Ev(ptr returned)
declare ptr @_ZN1SD1Ev(ptr returned)
declare i32 @_tlv_atexit(ptr, ptr, ptr)

; Make sure Epilog does not overwrite an explicitly-handled CSR in CXX_FAST_TLS.
; THUMB-LABEL: _ZTW2sg
; THUMB: push {{.*}}lr
; THUMB: blx
; THUMB: bne{{(.w)?}} [[TH_end:.?LBB0_[0-9]+]]
; THUMB: blx
; THUMB: tlv_atexit
; THUMB: [[TH_end]]:
; THUMB: blx
; THUMB: r4
; THUMB: pop {{.*}}r4
define cxx_fast_tlscc nonnull ptr @_ZTW2sg() nounwind "frame-pointer"="all" {
  %.b.i = load i1, ptr @__tls_guard, align 1
  br i1 %.b.i, label %__tls_init.exit, label %init.i

init.i:
  store i1 true, ptr @__tls_guard, align 1
  %call.i.i = tail call ptr @_ZN1SC1Ev(ptr nonnull @sg)
  %1 = tail call i32 @_tlv_atexit(ptr nonnull @_ZN1SD1Ev, ptr nonnull @sg, ptr nonnull @__dso_handle)
  br label %__tls_init.exit

__tls_init.exit:
  ret ptr @sg
}

; CHECK-LABEL: _ZTW2sg
; CHECK: push {r4, r5, r7, lr}
; CHECK: push {r11, r12}
; CHECK-NOT: vpush {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-NOT: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK: blx
; CHECK: bne [[BB_end:.?LBB0_[0-9]+]]
; CHECK: blx
; CHECK: tlv_atexit
; CHECK: [[BB_end]]:
; CHECK: blx
; CHECK-NOT: vpop {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-NOT: vpop {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-NOT: pop {r9, r12}
; CHECK-NOT: pop {r1, r2, r3, r4, r7, pc}
; CHECK: pop {r4, r5, r7, pc}

; CHECK-O0-LABEL: _ZTW2sg
; WATCH-O0: push {r1, r2, r3, r6, r7, lr}
; IOS-O0: push {r1, r2, r3, r7, lr}
; CHECK-O0: push {r9, r12}
; CHECK-O0: vpush {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-O0: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-O0: blx
; CHECK-O0: bne [[BB_end:.?LBB0_[0-9]+]]
; CHECK-O0: blx
; CHECK-O0: tlv_atexit
; CHECK-O0: [[BB_end]]:
; CHECK-O0: blx
; CHECK-O0: vpop {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-O0: vpop {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-O0: pop {r9, r12}
; WATCH-O0: pop {r1, r2, r3, r6, r7, pc}
; IOS-O0: pop {r1, r2, r3, r7, pc}

; CHECK-LABEL: _ZTW4sum1
; CHECK-NOT: push {r1, r2, r3, r4, r7, lr}
; CHECK-NOT: push {r9, r12}
; CHECK-NOT: vpush {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-NOT: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK: blx

; CHECK-O0-LABEL: _ZTW4sum1
; CHECK-O0-NOT: vpush
; CHECK-O0-NOT: vstr
; CHECK-O0-NOT: vpop
; CHECK-O0-NOT: vldr
; CHECK-O0: pop
define cxx_fast_tlscc nonnull ptr @_ZTW4sum1() nounwind "frame-pointer"="all" {
  ret ptr @sum1
}

; Make sure at O0, we don't generate spilling/reloading of the CSRs.
; CHECK-O0-LABEL: tls_test2
; CHECK-O0: push
; CHECK-O0-NOT: vpush
; CHECK-O0-NOT: vstr
; CHECK-O0: tls_helper
; CHECK-O0-NOT: vpop
; CHECK-O0-NOT: vldr
; CHECK-O0: pop
declare cxx_fast_tlscc void @tls_helper()
define cxx_fast_tlscc ptr @tls_test2() #1 "frame-pointer"="all" {
  call cxx_fast_tlscc void @tls_helper()
  ret ptr @tC
}

; Make sure we do not allow tail call when caller and callee have different
; calling conventions.
declare ptr @_ZN1CD1Ev(ptr readnone returned %this)
; CHECK-LABEL: tls_test
; CHECK: bl __tlv_atexit
define cxx_fast_tlscc void @__tls_test() "frame-pointer"="all" {
entry:
  store i32 0, ptr @tC, align 4
  %0 = tail call i32 @_tlv_atexit(ptr @_ZN1CD1Ev, ptr @tC, ptr nonnull @__dso_handle) #1
  ret void
}

declare void @somefunc()
define cxx_fast_tlscc void @test_ccmismatch_notail() "frame-pointer"="all" {
; A tail call is not possible here because somefunc does not preserve enough
; registers.
; CHECK-LABEL: test_ccmismatch_notail:
; CHECK-NOT: b _somefunc
; CHECK: bl _somefunc
  tail call void @somefunc()
  ret void
}

declare cxx_fast_tlscc void @some_fast_tls_func()
define void @test_ccmismatch_tail() "frame-pointer"="all" {
; We can perform a tail call here because some_fast_tls_func preserves all
; necessary registers (and more).
; CHECK-LABEL: test_ccmismatch_tail:
; CHECK-NOT: bl _some_fast_tls_func
; CHECK: b _some_fast_tls_func
  tail call cxx_fast_tlscc void @some_fast_tls_func()
  ret void
}

attributes #0 = { nounwind "frame-pointer"="all" }
attributes #1 = { nounwind }
