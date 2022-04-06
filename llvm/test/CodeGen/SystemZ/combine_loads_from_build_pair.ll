; RUN: llc -verify-machineinstrs  -mtriple=s390x-linux-gnu < %s -debug -stop-after=machineverifier 2>&1 | FileCheck %s

; REQUIRES: asserts
define i128 @func1({ i128, i8* } %struct) {
; Verify that we get a combine on the build_pair, creating a LD8 load somewhere
; between "Initial selection DAG" and "Optimized lowered selection DAG".
; The target is big-endian, and stack grows towards higher addresses,
; so we expect the LD8 to load from the address used in the original HIBITS
; load.
; CHECK-LABEL: Initial selection DAG:
; CHECK:     [[LOBITS:t[0-9]+]]: i64,ch = load<(load (s64))>
; CHECK:     [[HIBITS:t[0-9]+]]: i64,ch = load<(load (s64))>
; CHECK: Combining: t{{[0-9]+}}: i128 = build_pair [[LOBITS]], [[HIBITS]]
; CHECK-NEXT: Creating new node
; CHECK-SAME: load<(load (s128), align 8)>
; CHECK-NEXT: into
; CHECK-SAME: load<(load (s128), align 8)>
; CHECK-LABEL: Optimized lowered selection DAG:
  %result = extractvalue {i128, i8* } %struct, 0
  ret i128 %result
}

