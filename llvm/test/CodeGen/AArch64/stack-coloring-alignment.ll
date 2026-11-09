; RUN: llc -filetype=null %s -mtriple=arm64-apple-darwin -misched-print-dags 2>&1 | FileCheck %s
; REQUIRES: asserts

; Test that stack coloring merging allocas of differing alignments does not
; corrupt BasicAA results.

define <3 x float> @test_stack_coloring_maintains_alignment_metadata() {
  ; CHECK-LABEL: test_stack_coloring_maintains_alignment_metadata:%bb.0
  ; CHECK:       SU(0): STRWui $wzr, %stack.0.alloca.a, 1 :: (store (s32) into %ir.alloca.b.1)
  ; CHECK:         Successors:
  ; CHECK-NEXT:      SU(1): Ord  Latency=1 Memory
  ; CHECK:       SU(1): %2:fpr128 = LDRQui %stack.0.alloca.a, 0 :: (dereferenceable load (s128) from %ir.alloca.a)
  ; CHECK:         Successors:
  ; CHECK-NEXT:      SU(2): Data Latency=2 Reg=%2
  %alloca.a = alloca [3 x float], align 4
  %alloca.b = alloca [3 x float], align 16

  call void @llvm.lifetime.start.p0(ptr %alloca.a)
  call void @escape(ptr %alloca.a)
  call void @llvm.lifetime.end.p0(ptr %alloca.a)

  call void @llvm.lifetime.start.p0(ptr %alloca.b)
  %alloca.b.1 = getelementptr i8, ptr %alloca.b, i64 4
  ; After merging stack slots, the following store must not be reordered past
  ; the subsequent load.
  store float 0., ptr %alloca.b.1
  %v = load <3 x float>, ptr %alloca.b, align 16
  call void @llvm.lifetime.end.p0(ptr %alloca.b)

  ret <3 x float> %v
}

declare void @escape(ptr)
