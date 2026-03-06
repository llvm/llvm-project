// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(offload-target-verifier{soft-check=true}))" --verify-diagnostics -split-input-file

// Test scalar i32 live-in value - should pass (scalars can be passed by value)
func.func @test_scalar_i32() {
  %alloca = memref.alloca() : memref<i32>
  %livein = memref.load %alloca[] : memref<i32>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %accalloca = memref.alloca() : memref<i32>
    memref.store %livein, %accalloca[] : memref<i32>
    acc.yield
  }
  return
}

// -----

// Test memref live-in without data clause - should fail
func.func @test_memref_f32() {
  // expected-note @below {{value}}
  %livein = memref.alloca() : memref<f32>
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.serial {
    %load = memref.load %livein[] : memref<f32>
    %accalloca = memref.alloca() : memref<f32>
    memref.store %load, %accalloca[] : memref<f32>
    acc.yield
  }
  return
}

// -----

// Test memref with copyin data clause - should pass
func.func @test_memref_f32_copyin() {
  %alloca = memref.alloca() : memref<f32>
  %livein = acc.copyin varPtr(%alloca : memref<f32>) -> memref<f32>
  // expected-remark @below {{passed validity check}}
  acc.serial dataOperands(%livein : memref<f32>) {
    %load = memref.load %livein[] : memref<f32>
    %accalloca = memref.alloca() : memref<f32>
    memref.store %load, %accalloca[] : memref<f32>
    acc.yield
  }
  return
}

// -----

// Test memref with private clause - should pass (privatized values are not live-in)
acc.private.recipe @privatization_memref_f32 : memref<f32> init {
^bb0(%arg0: memref<f32>):
  %0 = memref.alloca() : memref<f32>
  acc.yield %0 : memref<f32>
}

func.func @test_memref_f32_private() {
  %livein = memref.alloca() : memref<f32>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %private = acc.private varPtr(%livein : memref<f32>) recipe(@privatization_memref_f32) -> memref<f32>
    %load = memref.load %private[] : memref<f32>
    %accalloca = memref.alloca() : memref<f32>
    memref.store %load, %accalloca[] : memref<f32>
    acc.yield
  }
  return
}

// -----

// Test llvm.ptr live-in without data clause - should fail
func.func @test_llvmptr_f64() {
  %c1 = arith.constant 1 : i64
  // expected-note @below {{value}}
  %alloca = llvm.alloca %c1 x f64 : (i64) -> !llvm.ptr
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.serial {
    %c1_inner = arith.constant 1 : i64
    %load = llvm.load %alloca : !llvm.ptr -> f64
    %accalloca = llvm.alloca %c1_inner x f64 : (i64) -> !llvm.ptr
    llvm.store %load, %accalloca : f64, !llvm.ptr
    acc.yield
  }
  return
}

// -----

// Test global symbol without declare attribute - should fail
memref.global @global_array : memref<10xf32> = uninitialized

func.func @test_global_symbol_no_declare() {
  // expected-warning @below {{illegal symbol(s): global_array}}
  acc.serial {
    %livein = memref.get_global @global_array : memref<10xf32>
    %c0 = arith.constant 0 : index
    %loaded = memref.load %livein[%c0] : memref<10xf32>
    acc.yield
  }
  return
}

// -----

// Test memref with GPU address space (device data) - should pass
func.func @test_memref_gpu_address_space() {
  %alloca = memref.alloca() : memref<f32, #gpu.address_space<global>>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %load = memref.load %alloca[] : memref<f32, #gpu.address_space<global>>
    acc.yield
  }
  return
}

// -----

// Test global symbol with acc.declare attribute - should pass
memref.global @global_array_declared : memref<10xf32> = dense<0.0> {acc.declare = #acc.declare<dataClause = acc_create>}

func.func @test_global_symbol_with_declare() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %livein = memref.get_global @global_array_declared : memref<10xf32>
    acc.yield
  }
  return
}

// -----

// Test gpu.launch region (another OffloadRegionOpInterface)
func.func @test_gpu_launch() {
  %c1 = arith.constant 1 : index
  // expected-note @below {{value}}
  %alloca = memref.alloca() : memref<f32>
  // expected-warning @below {{1 illegal live-in value(s)}}
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %load = memref.load %alloca[] : memref<f32>
    gpu.terminator
  }
  return
}

// -----

// Test acc.parallel region
func.func @test_acc_parallel() {
  // expected-note @below {{value}}
  %alloca = memref.alloca() : memref<f32>
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.parallel {
    %load = memref.load %alloca[] : memref<f32>
    acc.yield
  }
  return
}

// -----

// Test acc.kernels region
func.func @test_acc_kernels() {
  // expected-note @below {{value}}
  %alloca = memref.alloca() : memref<f32>
  // expected-warning @below {{1 illegal live-in value(s)}}
  acc.kernels {
    %load = memref.load %alloca[] : memref<f32>
    acc.terminator
  }
  return
}

// -----

// Test device global (memref.global with GPU address space) - should pass
memref.global @device_global : memref<10xf32, #gpu.address_space<global>> = uninitialized

func.func @test_device_global() {
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %livein = memref.get_global @device_global : memref<10xf32, #gpu.address_space<global>>
    acc.yield
  }
  return
}

// -----

// Test complex scalar (complex types can be passed by value)
func.func @test_complex_scalar() {
  %alloca = memref.alloca() : memref<complex<f32>>
  %livein = memref.load %alloca[] : memref<complex<f32>>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %accalloca = memref.alloca() : memref<complex<f32>>
    memref.store %livein, %accalloca[] : memref<complex<f32>>
    acc.yield
  }
  return
}

// -----

// Test index type scalar
func.func @test_index_scalar() {
  %c10 = arith.constant 10 : index
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %c1 = arith.constant 1 : index
    %sum = arith.addi %c10, %c1 : index
    acc.yield
  }
  return
}

// -----

// Test f64 scalar
func.func @test_f64_scalar() {
  %alloca = memref.alloca() : memref<f64>
  %livein = memref.load %alloca[] : memref<f64>
  // expected-remark @below {{passed validity check}}
  acc.serial {
    %accalloca = memref.alloca() : memref<f64>
    memref.store %livein, %accalloca[] : memref<f64>
    acc.yield
  }
  return
}
