// RUN: mlir-opt %s -enable-arm-streaming="mode=locally enable-za" \
// RUN:   -convert-vector-to-arm-sme -convert-vector-to-llvm="enable-arm-sme" \
// RUN:   -allocate-arm-sme-tiles -test-lower-to-llvm | \
// RUN: mlir-translate -mlir-to-llvmir | \
// RUN: %lli_aarch64_cmd --march=aarch64 --mattr="+sve,+sme" \
// RUN:   --entry-function=za0_d_f64 \
// RUN:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s --check-prefix=CHECK-ZA0_D

// Integration test demonstrating load/store to/from SME ZA tile.

llvm.func @printF64(f64)
llvm.func @printOpen()
llvm.func @printClose()
llvm.func @printComma()
llvm.func @printNewline()

func.func @za0_d_f64() -> i32 {
  %c0 = arith.constant 0 : index
  %c0_f64 = arith.constant 0.0 : f64
  %c1_f64 = arith.constant 1.0 : f64
  %c1_index = arith.constant 1 : index

  %min_elts_d = arith.constant 2 : index
  %vscale = vector.vscale

  // "svl" refers to the Streaming Vector Length and "svl_d" the number of
  // 64-bit elements in a vector of SVL bits.
  %svl_d = arith.muli %min_elts_d, %vscale : index

  // Allocate "mem1" and fill each "row" with row number.
  //
  // For example, assuming an SVL of 256-bits:
  //
  //   0.1, 0.1, 0.1, 0.1
  //   1.1, 1.1, 1.1, 1.1
  //   2.1, 2.1, 2.1, 2.1
  //   3.1, 3.1, 3.1, 3.1
  //
  %tilesize = arith.muli %svl_d, %svl_d : index
  %mem1 = memref.alloca(%tilesize) : memref<?xf64>
  %init_0 = arith.constant 0.1 : f64
  scf.for %i = %c0 to %tilesize step %svl_d iter_args(%val = %init_0) -> (f64) {
    %splat_val = vector.broadcast %val : f64 to vector<[2]xf64>
    vector.store %splat_val, %mem1[%i] : memref<?xf64>, vector<[2]xf64>
    %val_next = arith.addf %val, %c1_f64 : f64
    scf.yield %val_next : f64
  }

  // Dump "mem1". The smallest SVL is 128-bits so the tile will be at least
  // 2x2xi64.
  //
  // CHECK-ZA0_D:      ( 0.1, 0.1
  // CHECK-ZA0_D-NEXT: ( 1.1, 1.1
  scf.for %i = %c0 to %tilesize step %svl_d {
    %tileslice = vector.load %mem1[%i] : memref<?xf64>, vector<[2]xf64>

    llvm.call @printOpen() : () -> ()
    scf.for %i2 = %c0 to %svl_d step %c1_index {
      %elem = vector.extractelement %tileslice[%i2 : index] : vector<[2]xf64>
      llvm.call @printF64(%elem) : (f64) -> ()
      %last_i = arith.subi %svl_d, %c1_index : index
      %isNotLastIter = arith.cmpi ult, %i2, %last_i : index
      scf.if %isNotLastIter {
        llvm.call @printComma() : () -> ()
      }
    }
    llvm.call @printClose() : () -> ()
    llvm.call @printNewline() : () -> ()
  }

  // Load ZA0.D from "mem1"
  %za0_d = vector.load %mem1[%c0] : memref<?xf64>, vector<[2]x[2]xf64>

  // Allocate "mem2" to store ZA0.D to
  %mem2 = memref.alloca(%tilesize) : memref<?xf64>

  // Zero "mem2"
  scf.for %i = %c0 to %tilesize step %c1_index {
    memref.store %c0_f64, %mem2[%i] : memref<?xf64>
  }

  // Verify "mem2" is zeroed by doing an add reduction with initial value of
  // zero
  %init_0_f64 = arith.constant 0.0 : f64
  %add_reduce = scf.for %vnum = %c0 to %tilesize step %svl_d iter_args(%iter = %init_0_f64) -> (f64) {
    %row = vector.load %mem2[%vnum] : memref<?xf64>, vector<[2]xf64>

    %inner_add_reduce = scf.for %offset = %c0 to %svl_d step %c1_index iter_args(%inner_iter = %init_0_f64) -> (f64) {
      %t = vector.extractelement %row[%offset : index] : vector<[2]xf64>
      %inner_add_reduce_next = arith.addf %inner_iter, %t : f64
      scf.yield %inner_add_reduce_next : f64
    }

    %add_reduce_next = arith.addf %iter, %inner_add_reduce : f64
    scf.yield %add_reduce_next : f64
  }

  // CHECK-ZA0_D: 0
  vector.print %add_reduce : f64

  // Dump zeroed "mem2". The smallest SVL is 128-bits so the tile will be at
  // least 2x2xi64.
  //
  // CHECK-ZA0_D-NEXT: ( 0, 0
  // CHECK-ZA0_D-NEXT: ( 0, 0
  scf.for %i = %c0 to %tilesize step %svl_d {
    %tileslice = vector.load %mem2[%i] : memref<?xf64>, vector<[2]xf64>

    llvm.call @printOpen() : () -> ()
    scf.for %i2 = %c0 to %svl_d step %c1_index {
      %elem = vector.extractelement %tileslice[%i2 : index] : vector<[2]xf64>
      llvm.call @printF64(%elem) : (f64) -> ()
      %last_i = arith.subi %svl_d, %c1_index : index
      %isNotLastIter = arith.cmpi ult, %i2, %last_i : index
      scf.if %isNotLastIter {
        llvm.call @printComma() : () -> ()
      }
    }
    llvm.call @printClose() : () -> ()
    llvm.call @printNewline() : () -> ()
  }

  // Verify "mem1" != "mem2"
  %init_1 = arith.constant 1 : i64
  %mul_reduce_0 = scf.for %vnum = %c0 to %tilesize step %svl_d iter_args(%iter = %init_1) -> (i64) {
    %row_1 = vector.load %mem1[%vnum] : memref<?xf64>, vector<[2]xf64>
    %row_2 = vector.load %mem2[%vnum] : memref<?xf64>, vector<[2]xf64>
    %cmp = arith.cmpf one, %row_1, %row_2 : vector<[2]xf64>

    %inner_mul_reduce = scf.for %i = %c0 to %svl_d step %c1_index iter_args(%inner_iter = %init_1) -> (i64) {
      %t = vector.extractelement %cmp[%i : index] : vector<[2]xi1>
      %t_i64 = arith.extui %t : i1 to i64
      %inner_mul_reduce_next = arith.muli %inner_iter, %t_i64 : i64
      scf.yield %inner_mul_reduce_next : i64
    }

    %mul_reduce_next = arith.muli %iter, %inner_mul_reduce : i64
    scf.yield %mul_reduce_next : i64
  }

  // CHECK-ZA0_D: 1
  vector.print %mul_reduce_0 : i64

  // Store ZA0.D to "mem2"
  vector.store %za0_d, %mem2[%c0] : memref<?xf64>, vector<[2]x[2]xf64>

  // Verify "mem1" == "mem2"
  %mul_reduce_1 = scf.for %vnum = %c0 to %tilesize step %svl_d iter_args(%iter = %init_1) -> (i64) {
    %row_1 = vector.load %mem1[%vnum] : memref<?xf64>, vector<[2]xf64>
    %row_2 = vector.load %mem2[%vnum] : memref<?xf64>, vector<[2]xf64>
    %cmp = arith.cmpf oeq, %row_1, %row_2 : vector<[2]xf64>

    %inner_mul_reduce = scf.for %i = %c0 to %svl_d step %c1_index iter_args(%inner_iter = %init_1) -> (i64) {
      %t = vector.extractelement %cmp[%i : index] : vector<[2]xi1>
      %t_i64 = arith.extui %t : i1 to i64
      %inner_mul_reduce_next = arith.muli %inner_iter, %t_i64 : i64
      scf.yield %inner_mul_reduce_next : i64
    }

    %mul_reduce_next = arith.muli %iter, %inner_mul_reduce : i64
    scf.yield %mul_reduce_next : i64
  }

  // CHECK-ZA0_D-NEXT: 1
  vector.print %mul_reduce_1 : i64

  // Dump "mem2". The smallest SVL is 128-bits so the tile will be at least
  // 2x2xi64.
  //
  // CHECK-ZA0_D-NEXT: ( 0.1, 0.1
  // CHECK-ZA0_D-NEXT: ( 1.1, 1.1
  scf.for %i = %c0 to %tilesize step %svl_d {
    %tileslice = vector.load %mem2[%i] : memref<?xf64>, vector<[2]xf64>

    llvm.call @printOpen() : () -> ()
    scf.for %i2 = %c0 to %svl_d step %c1_index {
      %elem = vector.extractelement %tileslice[%i2 : index] : vector<[2]xf64>
      llvm.call @printF64(%elem) : (f64) -> ()
      %last_i = arith.subi %svl_d, %c1_index : index
      %isNotLastIter = arith.cmpi ult, %i2, %last_i : index
      scf.if %isNotLastIter {
        llvm.call @printComma() : () -> ()
      }
    }
    llvm.call @printClose() : () -> ()
    llvm.call @printNewline() : () -> ()
  }

  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
