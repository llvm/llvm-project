; Check behavior of the -float-abi command-line option; it should
; synthesize the "float-abi" module flag, unless one is already
; present

; RUN: split-file %s %t

; -float-abi=hard writes the module flag.
; RUN: llc -mtriple=armv7-none-eabi -float-abi=hard -stop-after=finalize-isel %t/none.ll -o - | FileCheck %s --check-prefix=HARD

; -float-abi=soft writes the module flag.
; RUN: llc -mtriple=armv7-none-eabi -float-abi=soft -stop-after=finalize-isel %t/none.ll -o - | FileCheck %s --check-prefix=SOFT

; Without -float-abi, no flag is synthesized.
; RUN: llc -mtriple=armv7-none-eabi -stop-after=finalize-isel %t/none.ll -o - | FileCheck %s --check-prefix=NONE

; -float-abi matching an existing in-IR flag is accepted.
; RUN: llc -mtriple=armv7-none-eabi -float-abi=hard -stop-after=finalize-isel %t/hard.ll -o - | FileCheck %s --check-prefix=HARD

; -float-abi conflicting with an existing in-IR flag is an error.
; RUN: not llc -mtriple=armv7-none-eabi -float-abi=soft -stop-after=finalize-isel %t/hard.ll -filetype=null 2>&1 | FileCheck %s --check-prefix=CONFLICT

;--- none.ll
define void @f() {
  ret void
}
; HARD: !{i32 1, !"float-abi", !"hard"}
; SOFT: !{i32 1, !"float-abi", !"soft"}
; NONE-NOT: !"float-abi"

;--- hard.ll
define void @f() {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}
; CONFLICT: -float-abi=soft conflicts with the "float-abi" module flag "hard"
