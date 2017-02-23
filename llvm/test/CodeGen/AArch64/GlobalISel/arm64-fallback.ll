; RUN: not llc -O0 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -global-isel -global-isel-abort=0 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=FALLBACK
; RUN: llc -O0 -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -verify-machineinstrs %s -o %t.out 2> %t.err
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
; ERROR: unable to lower arguments: i128 (i128)* (in function: ABIi128)
; FALLBACK: ldr q0,
; FALLBACK-NEXT: bl __fixunstfti
;
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to lower arguments: i128 (i128)* (in function: ABIi128)
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

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate constant: [1 x double] (in function: constant)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for constant
; FALLBACK-WITH-REPORT-OUT-LABEL: constant:
; FALLBACK-WITH-REPORT-OUT: fmov d0, #1.0
define [1 x double] @constant() {
  ret [1 x double] [double 1.0]
}

  ; The key problem here is that we may fail to create an MBB referenced by a
  ; PHI. If so, we cannot complete the G_PHI and mustn't try or bad things
  ; happen.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: cannot select: G_STORE %vreg4, %vreg2; mem:ST4[%addr] GPR:%vreg4,%vreg2 (in function: pending_phis)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for pending_phis
; FALLBACK-WITH-REPORT-OUT-LABEL: pending_phis:
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

  ; General legalizer inability to handle types whose size wasn't a power of 2.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %vreg1<def>(s42) = G_LOAD %vreg0; mem:LD6[%addr](align=8) (in function: odd_type)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for odd_type
; FALLBACK-WITH-REPORT-OUT-LABEL: odd_type:
define void @odd_type(i42* %addr) {
  %val42 = load i42, i42* %addr
  ret void
}

  ; RegBankSelect crashed when given invalid mappings, and AArch64's
  ; implementation produce valid-but-nonsense mappings for G_SEQUENCE.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to map instruction: %vreg0<def>(s128) = G_SEQUENCE %vreg1, 0, %vreg2, 64; GPR:%vreg1,%vreg2 (in function: sequence_mapping)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for sequence_mapping
; FALLBACK-WITH-REPORT-OUT-LABEL: sequence_mapping:
define void @sequence_mapping([2 x i64] %in) {
  ret void
}

  ; Legalizer was asserting when it enountered an unexpected default action.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %vreg9<def>(s128) = G_INSERT %vreg10, %vreg0, 32; (in function: legal_default)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for legal_default
; FALLBACK-WITH-REPORT-LABEL: legal_default:
define void @legal_default([8 x i8] %in) {
  insertvalue { [4 x i8], [8 x i8], [4 x i8] } undef, [8 x i8] %in, 1
  ret void
}

  ; AArch64 was asserting instead of returning an invalid mapping for unknown
  ; sizes.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: ret: '  ret i128 undef' (in function: sequence_sizes)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for sequence_sizes
; FALLBACK-WITH-REPORT-LABEL: sequence_sizes:
define i128 @sequence_sizes([8 x i8] %in) {
  ret i128 undef
}

; Just to make sure we don't accidentally emit a normal load/store.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: cannot select: %vreg2<def>(s64) = G_LOAD %vreg0; mem:LD8[%addr] GPR:%vreg2,%vreg0 (in function: atomic_ops)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for atomic_ops
; FALLBACK-WITH-REPORT-LABEL: atomic_ops:
define i64 @atomic_ops(i64* %addr) {
  store atomic i64 0, i64* %addr unordered, align 8
  %res = load atomic i64, i64* %addr seq_cst, align 8
  ret i64 %res
}
