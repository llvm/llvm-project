; RUN: split-file %s %t
; RUN: opt -passes=verify -disable-output %t/valid.ll
; RUN: not opt -passes=verify -disable-output %t/instruction.ll 2>&1 | FileCheck %s --check-prefix=LOCATION
; RUN: not opt -passes=verify -disable-output %t/global.ll 2>&1 | FileCheck %s --check-prefix=LOCATION
; RUN: not opt -passes=verify -disable-output %t/short.ll 2>&1 | FileCheck %s --check-prefix=SHORT
; RUN: not opt -passes=verify -disable-output %t/type.ll 2>&1 | FileCheck %s --check-prefix=TYPE
; RUN: not opt -passes=verify -disable-output %t/duplicate.ll 2>&1 | FileCheck %s --check-prefix=TYPE
; RUN: sed 's/!0 !wave.profile !1/!1 !wave.profile !0/' %t/duplicate.ll | not opt -passes=verify -disable-output 2>&1 | FileCheck %s --check-prefix=TYPE
; RUN: not opt -passes=verify -disable-output %t/block-location.ll 2>&1 | FileCheck %s --check-prefix=BLOCK-LOCATION
; RUN: not opt -passes=verify -disable-output %t/block-short.ll 2>&1 | FileCheck %s --check-prefix=BLOCK-SHORT
; RUN: not opt -passes=verify -disable-output %t/block-type.ll 2>&1 | FileCheck %s --check-prefix=BLOCK-TYPE
; RUN: not opt -passes=verify -disable-output %t/declaration.ll 2>&1 | FileCheck %s --check-prefix=LOCATION
; RUN: not opt -passes=verify -disable-output %t/block-function.ll 2>&1 | FileCheck %s --check-prefix=BLOCK-LOCATION
; RUN: not opt -passes=verify -disable-output %t/block-global.ll 2>&1 | FileCheck %s --check-prefix=BLOCK-LOCATION

; LOCATION: wave.profile is only valid on function definitions
; SHORT: wave.profile requires a version, function ID, and wave counts
; TYPE: wave.profile operands must be i64
; BLOCK-LOCATION: wave.profile.block is only valid on terminators
; BLOCK-SHORT: wave.profile.block requires a version, function ID, block ID, and count-valid flag
; BLOCK-TYPE: wave.profile.block operands must be i64

;--- valid.ll
; Stale fingerprints, unknown versions, and stale block counts are ignored by
; consumers, not rejected by the verifier after an optimization changes IR.
define void @stale() !wave.profile !0 {
  ret void
}
!0 = !{i64 1, i64 0, i64 100, i64 200}

;--- instruction.ll
define void @bad() {
  ret void, !wave.profile !0
}
!0 = !{i64 1, i64 0, i64 100}

;--- global.ll
@bad = global i32 0, !wave.profile !0
!0 = !{i64 1, i64 0, i64 100}

;--- short.ll
define void @bad() !wave.profile !0 {
  ret void
}
!0 = !{i64 1, i64 0}

;--- type.ll
define void @bad() !wave.profile !0 {
  ret void
}
!0 = !{i64 1, i64 0, i32 100}

;--- duplicate.ll
define void @bad() !wave.profile !0 !wave.profile !1 {
  ret void
}
!0 = !{i64 1, i64 0, i32 100}
!1 = !{i64 1, i64 0, i64 100}

;--- block-location.ll
define void @bad() {
  %value = add i32 1, 2, !wave.profile.block !0
  ret void
}
!0 = !{i64 2, i64 0, i64 0, i64 1}

;--- block-short.ll
define void @bad() {
  ret void, !wave.profile.block !0
}
!0 = !{i64 2, i64 0, i64 0}

;--- block-type.ll
define void @bad() {
  ret void, !wave.profile.block !0
}
!0 = !{i64 2, i64 0, i32 0, i64 1}

;--- declaration.ll
declare !wave.profile !0 void @bad()
!0 = !{i64 2, i64 0, i64 100}

;--- block-function.ll
define void @bad() !wave.profile.block !0 {
  ret void
}
!0 = !{i64 2, i64 0, i64 0, i64 1}

;--- block-global.ll
@bad = global i32 0, !wave.profile.block !0
!0 = !{i64 2, i64 0, i64 0, i64 1}
