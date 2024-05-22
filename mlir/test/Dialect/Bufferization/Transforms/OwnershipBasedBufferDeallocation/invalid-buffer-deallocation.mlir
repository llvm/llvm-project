// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation -split-input-file %s


// Test Case: explicit control-flow loop with a dynamically allocated buffer.
// The BufferDeallocation transformation should fail on this explicit
// control-flow loop since they are not supported.

// expected-error@+1 {{Only structured control-flow loops are supported}}
func.func @loop_dynalloc(
  %arg0 : i32,
  %arg1 : i32,
  %arg2: memref<?xf32>,
  %arg3: memref<?xf32>) {
  %const0 = arith.constant 0 : i32
  cf.br ^loopHeader(%const0, %arg2 : i32, memref<?xf32>)

^loopHeader(%i : i32, %buff : memref<?xf32>):
  %lessThan = arith.cmpi slt, %i, %arg1 : i32
  cf.cond_br %lessThan,
    ^loopBody(%i, %buff : i32, memref<?xf32>),
    ^exit(%buff : memref<?xf32>)

^loopBody(%val : i32, %buff2: memref<?xf32>):
  %const1 = arith.constant 1 : i32
  %inc = arith.addi %val, %const1 : i32
  %size = arith.index_cast %inc : i32 to index
  %alloc1 = memref.alloc(%size) : memref<?xf32>
  cf.br ^loopHeader(%inc, %alloc1 : i32, memref<?xf32>)

^exit(%buff3 : memref<?xf32>):
  test.copy(%buff3, %arg3) : (memref<?xf32>, memref<?xf32>)
  return
}

// -----

// Test Case: explicit control-flow loop with a dynamically allocated buffer.
// The BufferDeallocation transformation should fail on this explicit
// control-flow loop since they are not supported.

// expected-error@+1 {{Only structured control-flow loops are supported}}
func.func @do_loop_alloc(
  %arg0 : i32,
  %arg1 : i32,
  %arg2: memref<2xf32>,
  %arg3: memref<2xf32>) {
  %const0 = arith.constant 0 : i32
  cf.br ^loopBody(%const0, %arg2 : i32, memref<2xf32>)

^loopBody(%val : i32, %buff2: memref<2xf32>):
  %const1 = arith.constant 1 : i32
  %inc = arith.addi %val, %const1 : i32
  %alloc1 = memref.alloc() : memref<2xf32>
  cf.br ^loopHeader(%inc, %alloc1 : i32, memref<2xf32>)

^loopHeader(%i : i32, %buff : memref<2xf32>):
  %lessThan = arith.cmpi slt, %i, %arg1 : i32
  cf.cond_br %lessThan,
    ^loopBody(%i, %buff : i32, memref<2xf32>),
    ^exit(%buff : memref<2xf32>)

^exit(%buff3 : memref<2xf32>):
  test.copy(%buff3, %arg3) : (memref<2xf32>, memref<2xf32>)
  return
}

// -----

func.func @free_effect() {
  %alloc = memref.alloc() : memref<2xi32>
  // expected-error @below {{memory free side-effect on MemRef value not supported!}}
  %new_alloc = memref.realloc %alloc : memref<2xi32> to memref<4xi32>
  return
}

// -----

func.func @free_effect() {
  %alloc = memref.alloc() : memref<2xi32>
  // expected-error @below {{memory free side-effect on MemRef value not supported!}}
  memref.dealloc %alloc : memref<2xi32>
  return
}

// -----

func.func @free_effect() {
  %true = arith.constant true
  %alloc = memref.alloc() : memref<2xi32>
  // expected-error @below {{No deallocation operations must be present when running this pass!}}
  bufferization.dealloc (%alloc : memref<2xi32>) if (%true)
  return
}
