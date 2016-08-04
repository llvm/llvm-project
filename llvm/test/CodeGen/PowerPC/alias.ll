; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -code-model=medium| FileCheck --check-prefix=CHECK --check-prefix=MEDIUM %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -code-model=large | FileCheck --check-prefix=CHECK --check-prefix=LARGE %s

@foo = global i32 42
@fooa = alias i32, i32* @foo

@foo2 = global i64 42
@foo2a = alias i64, i64* @foo2

; CHECK-LABEL: bar:
define i32 @bar() {
; MEDIUM: addis 3, 2, fooa@toc@ha
; LARGE: addis 3, 2, .L[[L0:.*]]@toc@ha
  %a = load i32, i32* @fooa
  ret i32 %a
}

; CHECK-LABEL: bar2:
define i64 @bar2() {
; MEDIUM: addis 3, 2, foo2a@toc@ha
; MEDIUM: addi 3, 3, foo2a@toc@l
; LARGE: addis 3, 2, .L[[L1:.*]]@toc@ha
  %a = load i64, i64* @foo2a
  ret i64 %a
}

; LARGE: .L[[L0]]:
; LARGE-NEXT: .tc fooa[TC],fooa

; LARGE: .L[[L1]]:
; LARGE-NEXT: .tc foo2a[TC],foo2a
