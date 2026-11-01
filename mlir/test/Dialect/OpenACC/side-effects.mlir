// RUN: mlir-opt %s --test-side-effects --verify-diagnostics --split-input-file

// acc.atomic.read declares its effects per operand: it reads the location
// designated by `x` and writes the location designated by `v`. Without them the
// operation reports no effects at all, which every memory analysis has to read
// as "unknown" and treat maximally conservatively -- in particular it cannot be
// seen to overwrite `v`.

func.func @atomic_read(%x: memref<i32>, %v: memref<i32>) {
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
  return
}

// -----

// The same effects are reported inside an acc.atomic.capture region. The
// capture itself carries RecursiveMemoryEffects and so does not implement
// MemoryEffectOpInterface; the test pass walks only operations that do, so the
// capture gets no remark of its own and the effects come from its body. The
// read writes `v` and reads `x`; the update both reads and writes `x`, which is
// what keeps `x` from being treated as overwritten.

func.func @atomic_capture(%x: memref<i32>, %v: memref<i32>) {
  acc.atomic.capture {
    // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
    // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
    // expected-remark @below {{found an instance of 'write' on op operand 0, on resource '<Default>'}}
    acc.atomic.update %x : memref<i32> {
    ^bb0(%arg0: i32):
      // expected-remark @below {{operation has no memory effects}}
      %0 = arith.constant 1 : i32
      // expected-remark @below {{operation has no memory effects}}
      %1 = arith.addi %arg0, %0 : i32
      acc.yield %1 : i32
    }
  }
  return
}
