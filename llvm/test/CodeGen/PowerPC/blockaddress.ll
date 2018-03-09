; RUN: llc -verify-machineinstrs < %s -code-model=small -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=SMALL
; RUN: llc -verify-machineinstrs < %s -code-model=medium -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=MEDIUM
; RUN: llc -verify-machineinstrs < %s -code-model=large -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=MEDIUM
; RUN: llc -verify-machineinstrs < %s -code-model=small -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=SMALL
; RUN: llc -verify-machineinstrs < %s -code-model=medium -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=MEDIUM
; RUN: llc -verify-machineinstrs < %s -code-model=large -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=MEDIUM

define i8* @test() {
entry:
  br label %here

here:                                             ; preds = %entry
; MEDIUM: .Ltmp[[TMP0:[0-9]+]]:
; MEDIUM: addis [[R0:[0-9]+]], 2, .LC[[LC0:[0-9]+]]@toc@ha
; MEDIUM: ld 3, .LC[[LC0]]@toc@l([[R0]])
; MEDIUM: blr
; MEDIUM: .LC[[LC0]]:
; MEDIUM: .tc .Ltmp[[TMP0]][TC],.Ltmp[[TMP0]]
; SMALL: .Ltmp[[TMP0:[0-9]+]]:
; SMALL: ld 3, .LC[[LC0:[0-9]+]]@toc(2)
; SMALL: blr
; SMALL: .LC[[LC0]]:
; SMALL: .tc .Ltmp[[TMP0]][TC],.Ltmp[[TMP0]]
  ret i8* blockaddress(@test, %here)
}

