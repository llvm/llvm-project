// DEFINE: %{entry_point} = entry
// DEFINE: %{compile} = mlir-opt %s -enable-arm-streaming="streaming-mode=streaming-locally za-mode=new-za" \
// DEFINE:   -convert-vector-to-arm-sme -convert-arm-sme-to-scf -allocate-arm-sme-tiles \
// DEFINE:   -convert-arm-sme-to-llvm -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:  -march=aarch64 -mattr=+sve,+sme \
// DEFINE:  -e %{entry_point} -entry-point-result=i32 \
// DEFINE:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib

// RUN: %{compile} | %{run} | FileCheck %s

func.func @entry() -> i32 {
  %c0 = arith.constant 0 : index
  %c1_i8 = arith.constant 1 : i8
  %c1_index = arith.constant 1 : index

  %c16 = arith.constant 16 : index
  %vscale = vector.vscale

  // "svl" refers to the Streaming Vector Length and "svl_b" the number of
  // 8-bit elements in a vector of SVL bits.
  %svl_b = arith.muli %c16, %vscale : index

  // Allocate memory and fill with ones.
  //
  // TODO: type conversion of rank > 1 vector types generates array(s) of
  // vectors. This is invalid for scalable vectors since LLVM doesn't support
  // arrays of scalable vectors. This prevents initializing 2-d vectors with
  // 'vector.store' or 'vector.transfer_write' ops until this is resolved or
  // there's a custom lowering path.
  %za_b = memref.alloca(%svl_b, %svl_b) : memref<?x?xi8>
  scf.for %i = %c0 to %svl_b step %c1_index {
    scf.for %j = %c0 to %svl_b step %c1_index {
      memref.store %c1_i8, %za_b[%i, %j] : memref<?x?xi8>
    }
  }

  // Verify memory is ones by doing a mul reduction with initial value of one.
  %init_1 = arith.constant 1 : i64
  %mul_reduce = scf.for %vnum = %c0 to %svl_b step %c1_index iter_args(%iter = %init_1) -> (i64) {
    %row = vector.load %za_b[%vnum, %c0] : memref<?x?xi8>, vector<[16]xi8>

    %inner_mul_reduce = scf.for %offset = %c0 to %svl_b step %c1_index iter_args(%inner_iter = %init_1) -> (i64) {
      %t = vector.extractelement %row[%offset : index] : vector<[16]xi8>
      %t_i64 = arith.extui %t : i8 to i64
      %inner_mul_reduce_next = arith.muli %inner_iter, %t_i64 : i64
      scf.yield %inner_mul_reduce_next : i64
    }

    %mul_reduce_next = arith.muli %iter, %inner_mul_reduce : i64
    scf.yield %mul_reduce_next : i64
  }

  // CHECK: 1
  vector.print %mul_reduce : i64

  // Verify the mul reduction works as expected.
  //
  // TODO: ZA currently isn't re-enabled after calls and is therefore disable
  // by the callee on return. Once this is resolved this can be moved to a
  // function.
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : i8
  %c7 = arith.constant 7 : index
  %c15 = arith.constant 15 : i8
  memref.store %c4, %za_b[%c3, %c7] : memref<?x?xi8>
  memref.store %c15, %za_b[%c7, %c3] : memref<?x?xi8>
  %mul_reduce2 = scf.for %vnum = %c0 to %svl_b step %c1_index iter_args(%iter = %init_1) -> (i64) {
    %row = vector.load %za_b[%vnum, %c0] : memref<?x?xi8>, vector<[16]xi8>

    %inner_mul_reduce = scf.for %offset = %c0 to %svl_b step %c1_index iter_args(%inner_iter = %init_1) -> (i64) {
      %t = vector.extractelement %row[%offset : index] : vector<[16]xi8>
      %t_i64 = arith.extui %t : i8 to i64
      %inner_mul_reduce_next = arith.muli %inner_iter, %t_i64 : i64
      scf.yield %inner_mul_reduce_next : i64
    }

    %mul_reduce_next = arith.muli %iter, %inner_mul_reduce : i64
    scf.yield %mul_reduce_next : i64
  }

  // 15*4=60
  // CHECK: 60
  vector.print %mul_reduce2 : i64

  // Fill memory with zeroes.
  //
  // This will get lowered to:
  //
  //   zero {za}
  //   for vnum = 0; vnum < SVLb; ++vnum;
  //     str za[vnum], [ptr]
  //     ...
  //
  %cst_0 = arith.constant dense<0> : vector<[16]x[16]xi8>
  vector.transfer_write %cst_0, %za_b[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>

  // Verify memory is zeroed by doing an add reduction with initial value of
  // zero.
  %init_0 = arith.constant 0 : i8
  %add_reduce = scf.for %vnum = %c0 to %svl_b step %c1_index iter_args(%iter = %init_0) -> (i8) {
    %row = vector.load %za_b[%vnum, %c0] : memref<?x?xi8>, vector<[16]xi8>
    %row_sum = vector.reduction <add>, %row : vector<[16]xi8> into i8
    %add_reduce_next = arith.addi %iter, %row_sum : i8
    scf.yield %add_reduce_next : i8
  }

  // CHECK-NEXT: 0
  vector.print %add_reduce : i8

  // Verify the add reduction works as expected.
  //
  // TODO: ZA currently isn't re-enabled after calls and is therefore disable
  // by the callee on return. Once this is resolved this can be moved to a
  // function.
  memref.store %c4, %za_b[%c3, %c7] : memref<?x?xi8>
  memref.store %c15, %za_b[%c7, %c3] : memref<?x?xi8>
  %add_reduce2 = scf.for %vnum = %c0 to %svl_b step %c1_index iter_args(%iter = %init_0) -> (i8) {
    %row = vector.load %za_b[%vnum, %c0] : memref<?x?xi8>, vector<[16]xi8>
    %row_sum = vector.reduction <add>, %row : vector<[16]xi8> into i8
    %add_reduce_next = arith.addi %iter, %row_sum : i8
    scf.yield %add_reduce_next : i8
  }

  // 15+4=19
  // CHECK-NEXT: 19
  vector.print %add_reduce2 : i8

  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
