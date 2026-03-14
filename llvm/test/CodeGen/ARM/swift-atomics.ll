; RUN: llc -mtriple=armv7-apple-ios6.0 -mcpu=swift < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios6.0 < %s | FileCheck %s --check-prefix=CHECK-STRICT-ATOMIC

; Release operations only need the store barrier provided by a "dmb ishst",

define void @test_store_release(ptr %p, i32 %v) {
; CHECK-LABEL: test_store_release:
; CHECK: dmb ishst
; CHECK: str

; CHECK-STRICT-ATOMIC-LABEL: test_store_release:
; CHECK-STRICT-ATOMIC: dmb {{ish$}}
  store atomic i32 %v, ptr %p release, align 4
  ret void
}

; However, if sequential consistency is needed *something* must ensure a release
; followed by an acquire does not get reordered. In that case a "dmb ishst" is
; not adequate.
define i32 @test_seq_cst(ptr %p, i32 %v) {
; CHECK-LABEL: test_seq_cst:
; CHECK: dmb ishst
; CHECK: str
; CHECK: dmb {{ish$}}
; CHECK: ldr
; CHECK: dmb {{ish$}}

; CHECK-STRICT-ATOMIC-LABEL: test_seq_cst:
; CHECK-STRICT-ATOMIC: dmb {{ish$}}
; CHECK-STRICT-ATOMIC: str
; CHECK-STRICT-ATOMIC: dmb {{ish$}}
; CHECK-STRICT-ATOMIC: ldr
; CHECK-STRICT-ATOMIC: dmb {{ish$}}

  store atomic i32 %v, ptr %p seq_cst, align 4
  %val = load atomic i32, ptr %p seq_cst, align 4
  ret i32 %val
}

; Also, pure acquire operations should definitely not have an ishst barrier.

define i32 @test_acq(ptr %addr) {
; CHECK-LABEL: test_acq:
; CHECK: ldr
; CHECK: dmb {{ish$}}

; CHECK-STRICT-ATOMIC-LABEL: test_acq:
; CHECK-STRICT-ATOMIC: dmb {{ish$}}
  %val = load atomic i32, ptr %addr acquire, align 4
  ret i32 %val
}
