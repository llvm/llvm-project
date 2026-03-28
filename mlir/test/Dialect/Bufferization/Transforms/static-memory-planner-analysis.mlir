// RUN: mlir-opt %s -split-input-file -verify-diagnostics \
// RUN:     --static-memory-planner-analysis


// -----

// Test 1: Two sequential non-overlapping alloc/dealloc pairs — both eligible.
//
// Block op indices (zero-based):
//   0: %alloc0 = memref.alloc
//   1: memref.dealloc %alloc0
//   2: %alloc1 = memref.alloc
//   3: memref.dealloc %alloc1
//   4: return
//
// alloc0 interval = [0, 1] — dealloc is lastUse, no other users.
// alloc1 interval = [2, 3] — dealloc is lastUse, no other users.
// The two intervals are non-overlapping: a future planner can reuse the same
// static region for both buffers.

func.func @simple_sequential() {
  // expected-remark @below {{static-memory-planner: eligible: size=4096 bytes, interval=[0,1]}}
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  // expected-remark @below {{static-memory-planner: eligible: size=2048 bytes, interval=[2,3]}}
  %alloc1 = memref.alloc() : memref<512xf32>
  memref.dealloc %alloc1 : memref<512xf32>
  return
}

// -----

// Test 2: Dynamic-shape allocation — skipped.
// The alloc has a runtime dimension (%n), so the pass cannot compute a static
// size or reason about it conservatively for static planning.

func.func @dynamic_shape_skipped(%n: index) {
  // expected-remark @below {{static-memory-planner: skip: dynamic shape}}
  %alloc = memref.alloc(%n) : memref<?xf32>
  return
}

// -----

// Test 3: No same-block dealloc — skipped.
// The alloc is unmatched (no memref.dealloc in any block), so the pass
// cannot establish a lifetime interval.

func.func @no_dealloc_skipped() {
  // expected-remark @below {{static-memory-planner: skip: no unique same-block dealloc}}
  %alloc = memref.alloc() : memref<1024xf32>
  return
}

// -----

// Test 4: Alloc inside scf.if (conditional) — skipped.
// Allocations nested inside conditionals are excluded from v1 scope because
// their liveness depends on runtime predicate evaluation.

func.func @conditional_alloc_skipped(%cond: i1) {
  scf.if %cond {
    // expected-remark @below {{static-memory-planner: skip: nested in loop or conditional}}
    %alloc = memref.alloc() : memref<1024xf32>
    memref.dealloc %alloc : memref<1024xf32>
    scf.yield
  }
  return
}

// -----

// Test 5: Alloc inside scf.for (loop) — skipped.
// Allocations inside loop bodies may execute a dynamic number of times;
// static planning requires reasoning outside loop nests.

func.func @loop_alloc_skipped(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    // expected-remark @below {{static-memory-planner: skip: nested in loop or conditional}}
    %alloc = memref.alloc() : memref<1024xf32>
    memref.dealloc %alloc : memref<1024xf32>
    scf.yield
  }
  return
}

// -----

// Test 6: expand_shape alias — alloc remains eligible and the alias is tracked.
//
// BufferViewFlowAnalysis::resolve(%alloc) finds both %alloc and %view.
// Users of %alloc (non-dealloc): expand_shape at idx 1.
// Users of %view: none.
// endIdx = max(deallocIdx=2, lastAliasUse=1) = 2.
// Interval = [0, 2].
//
// This test verifies that the presence of a derived alias does not cause
// misclassification or a crash.

func.func @view_alias_tracked() {
  // expected-remark @below {{static-memory-planner: eligible: size=4096 bytes, interval=[0,2]}}
  %alloc = memref.alloc() : memref<1024xf32>
  %view = memref.expand_shape %alloc [[0, 1]] output_shape [2, 512]
          : memref<1024xf32> into memref<2x512xf32>
  memref.dealloc %alloc : memref<1024xf32>
  return
}
