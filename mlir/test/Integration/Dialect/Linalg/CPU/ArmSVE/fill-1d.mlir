// RUN: mlir-opt %s -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule -lower-vector-mask -one-shot-bufferize -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd -e=entry -entry-point-result=void --march=aarch64 --mattr="+sve" -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @printTestEnd() {
  %0 = llvm.mlir.addressof @str_sve_end : !llvm.ptr<array<24 x i8>>
  %1 = llvm.mlir.constant(0 : index) : i64
  %2 = llvm.getelementptr %0[%1, %1]
    : (!llvm.ptr<array<24 x i8>>, i64, i64) -> !llvm.ptr<i8>
  llvm.call @printCString(%2) : (!llvm.ptr<i8>) -> ()
  return
}

func.func @entry() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %step = arith.constant 1 : index
  %c1_f32 = arith.constant 123.0 : f32

  %vscale = vector.vscale
  %vl_fp = arith.muli %c4, %vscale : index
  %vec = bufferization.alloc_tensor(%vl_fp) : tensor<?xf32>

  %vec_out = scf.for %i = %c0 to %vl_fp step %step iter_args(%vin = %vec) -> tensor<?xf32> {
    %vout = tensor.insert %c1_f32 into %vin[%i] : tensor<?xf32>
    scf.yield %vout : tensor<?xf32>
  }

  %pi = arith.constant  3.14 : f32
  %vec_out_1 = linalg.fill ins(%pi : f32) outs(%vec_out : tensor<?xf32>) -> tensor<?xf32>

  // There are at least 4 f32 elements in every SVE vector. For implementations
  // with wider vectors, you should see more elements being printed.
  // CHECK: 3.14
  // CHECK: 3.14
  // CHECK: 3.14
  // CHECK: 3.14
  scf.for %i = %c0 to %vl_fp step %step {
    %element = tensor.extract %vec_out_1[%i] : tensor<?xf32>
    vector.print %element : f32
  }

  // CHECK: SVE: END OF TEST OUTPUT
  func.call @printTestEnd() : () -> ()

  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.vectorize %0 vector_sizes [[4]] : !transform.any_op
}

llvm.func @printCString(!llvm.ptr<i8>)
llvm.mlir.global internal constant @str_sve_end("SVE: END OF TEST OUTPUT\0A")
