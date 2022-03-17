// RUN: export M=24 && export K=64 && export N=192 && export ITERS=10 && \
// RUN: cat %s | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g'| \
// RUN: mlir-opt -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul register-tile-sizes=12,32,16 vectorize" | \
// RUN: mlir-opt -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.fill register-tile-sizes=4,32 vectorize" | \
// RUN: mlir-opt -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=memref.copy register-tile-sizes=4,32 vectorize" | \

// RUN: mlir-opt -pass-pipeline="func.func(canonicalize,convert-vector-to-scf,lower-affine,convert-linalg-to-loops)" | \
// RUN: mlir-opt -pass-pipeline="func.func(canonicalize,convert-scf-to-cf),convert-vector-to-llvm,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts" | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// Activate to dump assembly
// R_UN:   -dump-object-file -object-filename=/tmp/a.o \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// Use tee to both print to stderr and FileCheck
// RUN: tee -a /dev/stderr | FileCheck %s


!elem_type_a = type f32
!elem_type_b = type f32
!elem_type_c = type f32
!row_major_A = type memref<${M}x${K}x!elem_type_a>
!row_major_B = type memref<${K}x${N}x!elem_type_b>
!row_major_C = type memref<${M}x${N}x!elem_type_c>

func @matmul(%a: !row_major_A, %b: !row_major_B, %c: !row_major_C)
// TODO: activate manually for now.
// attributes { passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]}
{
  linalg.matmul ins(%a, %b : !row_major_A, !row_major_B)
    outs(%c: !row_major_C)
  return
}

func @print_perf(%iters: index, %total_time: f64) {
  %c2 = arith.constant 2 : index
  %cM = arith.constant ${M} : index
  %cN = arith.constant ${N} : index
  %cK = arith.constant ${K} : index

  %mn = arith.muli %cM, %cN : index
  %mnk = arith.muli %mn, %cK : index

  // 2*M*N*K.
  %flops_per_iter = arith.muli %c2, %mnk : index
  %flops = arith.muli %iters, %flops_per_iter : index
  %flops_i64 = arith.index_cast %flops : index to i64
  %flops_f = arith.sitofp %flops_i64 : i64 to f64
  %flops_per_s = arith.divf %flops_f, %total_time : f64
  vector.print %flops_per_s : f64

  return
}

func @main() {
  %v0 = arith.constant 0.0 : !elem_type_a
  %v1 = arith.constant 1.0 : !elem_type_a

  %A = memref.alloc() : !row_major_A
  %B = memref.alloc() : !row_major_B
  %C = memref.alloc() : !row_major_C

  linalg.fill ins(%v1 : !elem_type_a) outs(%A : !row_major_A)
  linalg.fill ins(%v1 : !elem_type_b) outs(%B : !row_major_B)
  linalg.fill ins(%v0 : !elem_type_c) outs(%C : !row_major_C)

  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %iters = arith.constant ${ITERS}: index

  /// Run and dump performance for matmul.
  /// Preheating run:
  scf.for %arg0 = %c0 to %iters step %c1 {
    %z = arith.constant 0.0 : !elem_type_c
    linalg.fill ins(%z : !elem_type_c) outs(%C : !row_major_C)
    call @matmul(%A, %B, %C) : (!row_major_A, !row_major_B, !row_major_C) -> ()
  }
  %t_start_matmul = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    // linalg.matmul writes %C in place, need to reset it to zero every time.
    // This is accounts for about 10-15% perf hit on small sizes.
    // Once linalg on tensors is ready, fusing fill at the register level will
    // be easy.
    %z = arith.constant 0.0 : !elem_type_c
    linalg.fill ins(%z : !elem_type_c) outs(%C : !row_major_C)
    call @matmul(%A, %B, %C) : (!row_major_A, !row_major_B, !row_major_C) -> ()
  }
  %t_end_matmul = call @rtclock() : () -> f64
  %tmatmul = arith.subf %t_end_matmul, %t_start_matmul: f64
  call @print_perf(%iters, %tmatmul) : (index, f64) -> ()

  // CHECK: {{^0$}}
  %C_ref = memref.alloc() : !row_major_C
  linalg.fill ins(%v0 : !elem_type_c) outs(%C_ref : !row_major_C)
  linalg.matmul ins(%A, %B : !row_major_A, !row_major_B)
    outs(%C_ref: !row_major_C)
  %act = memref.cast %C : !row_major_C to memref<*xf32>
  %exp = memref.cast %C_ref : !row_major_C to memref<*xf32>
  %errors = call @verifyMemRefF32(%act, %exp) : (memref<*xf32>, memref<*xf32>) -> i64
  vector.print %errors : i64
  memref.dealloc %C_ref : !row_major_C

  memref.dealloc %A : !row_major_A
  memref.dealloc %B : !row_major_B
  memref.dealloc %C : !row_major_C

  return
}

func private @rtclock() -> f64
func private @verifyMemRefF32(memref<*xf32>, memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }

// TODO: init with random, run and check output.
// func private @fill_random_f32(memref<*xf32>)
