; RUN: not llc -O0 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -global-isel -global-isel-abort=0 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=FALLBACK
; RUN: llc -O0 -global-isel -global-isel-abort=2 -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err
; This file checks that the fallback path to selection dag works.
; The test is fragile in the sense that it must be updated to expose
; something that fails with global-isel.
; When we cannot produce a test case anymore, that means we can remove
; the fallback path.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--"

; We use __fixunstfti as the common denominator for __fixunstfti on Linux and
; ___fixunstfti on iOS
; ERROR: Unable to lower arguments
; FALLBACK: ldr q0,
; FALLBACK-NEXT: bl __fixunstfti
;
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for ABIi128
; FALLBACK-WITH-REPORT-OUT-LABEL: ABIi128:
; FALLBACK-WITH-REPORT-OUT: ldr q0,
; FALLBACK-WITH-REPORT-OUT-NEXT: bl __fixunstfti
define i128 @ABIi128(i128 %arg1) {
  %farg1 =       bitcast i128 %arg1 to fp128
  %res = fptoui fp128 %farg1 to i128
  ret i128 %res
}

; It happens that we don't handle ConstantArray instances yet during
; translation. Any other constant would be fine too.

; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for constant
; FALLBACK-WITH-REPORT-OUT-LABEL: constant:
; FALLBACK-WITH-REPORT-OUT: fmov d0, #1.0
define [1 x double] @constant() {
  ret [1 x double] [double 1.0]
}

  ; The key problem here is that we may fail to create an MBB referenced by a
  ; PHI. If so, we cannot complete the G_PHI and mustn't try or bad things
  ; happen.
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for pending_phis
define i32 @pending_phis(i1 %tst, i32 %val, i32* %addr) {
  br i1 %tst, label %true, label %false

end:
  %res = phi i32 [%val, %true], [42, %false]
  ret i32 %res

true:
  store atomic i32 42, i32* %addr seq_cst, align 4
  br label %end

false:
  br label %end

}
