; RUN: llc -mtriple=hexagon -hexagon-small-data-threshold=8 < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -hexagon-small-data-threshold=0 < %s \
; RUN:   | FileCheck --check-prefix=NOSDATA %s

; Test coverage for HexagonTargetObjectFile: exercise the small data/BSS
; section selection logic.

@small_global = global i32 0, align 4
@large_global = global [256 x i32] zeroinitializer, align 4
@small_init = global i32 7, align 4

; With small data threshold=8, small globals go into .sdata/.sbss.
; CHECK-LABEL: test_small_data:
; CHECK: memw(gp+#small_global)
; NOSDATA-LABEL: test_small_data:
; NOSDATA: memw(##small_global)
define i32 @test_small_data() {
entry:
  %val = load i32, ptr @small_global, align 4
  ret i32 %val
}

; Large arrays should NOT go into small data.
; CHECK-LABEL: test_large_data:
; CHECK: ##large_global
define i32 @test_large_data() {
entry:
  %val = load i32, ptr @large_global, align 4
  ret i32 %val
}

; Initialized small data should also be accessible via GP.
; CHECK-LABEL: test_init:
; CHECK: memw(gp+#small_init)
define i32 @test_init() {
entry:
  %val = load i32, ptr @small_init, align 4
  ret i32 %val
}

