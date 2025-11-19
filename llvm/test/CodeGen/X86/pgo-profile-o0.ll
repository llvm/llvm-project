; RUN: llc -mtriple=x86_64-- -O0 -pgo-kind=pgo-sample-use-pipeline -debug-pass=Structure %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=PASSES
; RUN: llc -mtriple=x86_64-- -O0 -pgo-kind=pgo-sample-use-pipeline -debug-only=branch-prob %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BRANCH_PROB
; RUN: llc -mtriple=x86_64-- -O0 -pgo-kind=pgo-sample-use-pipeline -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR

; REQUIRES: asserts

; This test verifies that PGO profile information (branch weights) is preserved
; during instruction selection at -O0.

; Test function with explicit branch weights from PGO.
define i32 @test_pgo_preservation(i32 %x) !prof !15 {
entry:
  %cmp = icmp sgt i32 %x, 10
  ; This branch has bias: 97 taken vs 3 not taken
  br i1 %cmp, label %if.then, label %if.else, !prof !16

if.then:
  ; Hot path - should have high frequency
  %add = add nsw i32 %x, 100
  br label %if.end

if.else:
  ; Cold path - should have low frequency
  %sub = sub nsw i32 %x, 50
  br label %if.end

if.end:
  %result = phi i32 [ %add, %if.then ], [ %sub, %if.else ]
  ret i32 %result
}

; Profile metadata with branch weights 97:3.
!15 = !{!"function_entry_count", i64 100}
!16 = !{!"branch_weights", i32 97, i32 3}

; Verify that Branch Probability Analysis runs at O0.
; PASSES: Branch Probability Analysis

; Verify that the branch probabilities reflect the exact profile data.
; BRANCH_PROB: ---- Branch Probability Info : test_pgo_preservation ----
; BRANCH_PROB: set edge entry -> 0 successor probability to {{.*}} = 97.00%
; BRANCH_PROB: set edge entry -> 1 successor probability to {{.*}} = 3.00%

; Verify that machine IR preserves the branch probabilities from profile data
; MIR: bb.0.entry:
; MIR-NEXT: successors: %bb.{{[0-9]+}}({{0x03d70a3d|0x7c28f5c3}}), %bb.{{[0-9]+}}({{0x7c28f5c3|0x03d70a3d}})
; The two successor probability values should be:
; - 0x7c28f5c3: approximately 97% (high probability successor)
; - 0x03d70a3d: approximately 3% (low probability successor)
