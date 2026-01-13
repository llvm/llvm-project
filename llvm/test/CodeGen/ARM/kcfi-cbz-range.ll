; RUN: llc -mtriple=thumbv7-linux-gnueabi -filetype=obj < %s
; RUN: llc -mtriple=thumbv7-linux-gnueabi < %s | FileCheck %s

; This test verifies that KCFI instrumentation doesn't cause "out of range
; pc-relative fixup value" errors when generating object files.
;
; The test creates a scenario with enough KCFI-instrumented indirect calls
; (~32 bytes each) that would push a cbz/cbnz instruction out of its ±126 byte
; range if the KCFI_CHECK pseudo-instruction size is not properly accounted for.
;
; Without the fix (KCFI_CHECK returns size 0):
;   - Backend thinks KCFI checks take no space
;   - Generates cbz to branch over the code
;   - During assembly, cbz target is >126 bytes away
;   - Assembly fails with "error: out of range pc-relative fixup value"
;
; With the fix (KCFI_CHECK returns size 32 for Thumb2):
;   - Backend correctly accounts for KCFI check expansion
;   - Avoids cbz or uses longer-range branch instructions
;   - Assembly succeeds, object file is generated

declare void @external_function(i32)

; Test WITHOUT KCFI: should generate cbz since calls are small
; CHECK-LABEL: test_without_kcfi:
; CHECK: cbz
; CHECK-NOT: bic{{.*}}#1
define i32 @test_without_kcfi(ptr %callback, i32 %x) {
entry:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %if_zero, label %if_nonzero

if_nonzero:
  ; Regular (non-KCFI) indirect calls - much smaller
  call void %callback()
  call void %callback()
  call void %callback()
  call void %callback()
  call void %callback()
  call void %callback()

  call void @external_function(i32 %x)
  %add1 = add i32 %x, 1
  ret i32 %add1

if_zero:
  call void @external_function(i32 0)
  ret i32 0
}

; Test WITH KCFI: should NOT generate cbz due to large KCFI checks
; CHECK-LABEL: test_with_kcfi:
; CHECK-NOT: cbz
; CHECK: bic{{.*}}#1
define i32 @test_with_kcfi(ptr %callback, i32 %x) !kcfi_type !1 {
entry:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %if_zero, label %if_nonzero

if_nonzero:
  ; Six KCFI-instrumented indirect calls (~192 bytes total, exceeds cbz range)
  call void %callback() [ "kcfi"(i32 12345678) ]
  call void %callback() [ "kcfi"(i32 12345678) ]
  call void %callback() [ "kcfi"(i32 12345678) ]
  call void %callback() [ "kcfi"(i32 12345678) ]
  call void %callback() [ "kcfi"(i32 12345678) ]
  call void %callback() [ "kcfi"(i32 12345678) ]

  ; Regular call to prevent optimization
  call void @external_function(i32 %x)
  %add1 = add i32 %x, 1
  ret i32 %add1

if_zero:
  call void @external_function(i32 0)
  ret i32 0
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
