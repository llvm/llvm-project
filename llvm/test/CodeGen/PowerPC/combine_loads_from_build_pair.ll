; RUN: llc -verify-machineinstrs -O0 -mcpu=g4 -mtriple=powerpc-apple-darwin8 < %s -debug -stop-after=machineverifier 2>&1 | FileCheck %s

; REQUIRES: asserts

define i64 @func1(i64 %p1, i64 %p2, i64 %p3, i64 %p4, { i64, i8* } %struct) {
; Verify that we get a combine on the build_pair, creating a LD8 load somewhere
; between "Initial selection DAG" and "Optimized lowered selection DAG".
; The target is big-endian, and stack grows towards higher addresses,
; so we expect the LD8 to load from the address used in the original HIBITS
; load.
; CHECK-LABEL: Initial selection DAG:
; CHECK-DAG:     [[LOBITS:t[0-9]+]]: i32,ch = load<(load 4 from %fixed-stack.1)>
; CHECK-DAG:     [[HIBITS:t[0-9]+]]: i32,ch = load<(load 4 from %fixed-stack.2)>
; CHECK: Combining: t{{[0-9]+}}: i64 = build_pair [[LOBITS]], [[HIBITS]]
; CHECK-NEXT: Creating new node
; CHECK-SAME: load<(load 8 from %fixed-stack.2, align 4)>
; CHECK-NEXT: into
; CHECK-SAME: load<(load 8 from %fixed-stack.2, align 4)>
; CHECK-LABEL: Optimized lowered selection DAG:
  %result = extractvalue {i64, i8* } %struct, 0
  ret i64 %result
}

