// RUN: export M=24 && export K=64 && export N=192 && export ITERS=10 && \
// RUN: cat %s | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g'| \
// RUN: mlir-opt -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul_i8_i8_i32 register-tile-sizes=12,32,16 vectorize" | \
// RUN: mlir-opt -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.fill register-tile-sizes=4,32 vectorize" | \
// RUN: mlir-opt -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.copy register-tile-sizes=4,32 vectorize" | \
// RUN: mlir-opt -canonicalize -convert-vector-to-scf -lower-affine -convert-linalg-to-loops | \

// RUN: mlir-opt -canonicalize -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm -mlir-disable-threading | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// Activate to dump assembly
// R_UN:   -dump-object-file -object-filename=/tmp/a.o \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// Use tee to both print to stderr and FileCheck
// RUN: tee -a /dev/stderr | FileCheck %s


!elem_type_a = type i8
!elem_type_b = type i8
!elem_type_c = type i32
!row_major_A = type memref<${M}x${K}x!elem_type_a>
!row_major_B = type memref<${K}x${N}x!elem_type_b>
!row_major_C = type memref<${M}x${N}x!elem_type_c>

func @matmul(%a: !row_major_A, %b: !row_major_B, %c: !row_major_C)
// TODO: activate manually for now.
// attributes { passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]}
{
  linalg.matmul_i8_i8_i32 ins(%a, %b : !row_major_A, !row_major_B)
    outs(%c: !row_major_C)
  return
}

func @print_perf(%iters: index, %total_time: f64) {
  %c2 = constant 2 : index
  %cM = constant ${M} : index
  %cN = constant ${N} : index
  %cK = constant ${K} : index

  %mn = muli %cM, %cN : index
  %mnk = muli %mn, %cK : index

  // 2*M*N*K.
  %flops_per_iter = muli %c2, %mnk : index
  %flops = muli %iters, %flops_per_iter : index
  %flops_i64 = index_cast %flops : index to i64
  %flops_f = sitofp %flops_i64 : i64 to f64
  %flops_per_s = divf %flops_f, %total_time : f64
  vector.print %flops_per_s : f64

  return
}

func @main() {
  %v0 = constant 0 : !elem_type_c
  %v1 = constant 1 : !elem_type_a

  %A = alloc() : !row_major_A
  %B = alloc() : !row_major_B
  %C = alloc() : !row_major_C

  linalg.fill(%A, %v1) : !row_major_A, !elem_type_a
  linalg.fill(%B, %v1) : !row_major_B, !elem_type_b
  linalg.fill(%C, %v0) : !row_major_C, !elem_type_c

  %c0 = constant 0: index
  %c1 = constant 1: index
  %iters = constant 100: index

  /// Run and dump performance for matmul.
  /// Preheating run:
  scf.for %arg0 = %c0 to %iters step %c1 {
    linalg.fill(%C, %v0) : !row_major_C, !elem_type_c
    call @matmul(%A, %B, %C) : (!row_major_A, !row_major_B, !row_major_C) -> ()
  }
  %t_start_matmul = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    // linalg.matmul writes %C in place, need to reset it to zero every time.
    // This is accounts for about 10-15% perf hit on small sizes.
    // Once linalg on tensors is ready, fusing fill at the register level will
    // be easy.
    linalg.fill(%C, %v0) : !row_major_C, !elem_type_c
    call @matmul(%A, %B, %C) : (!row_major_A, !row_major_B, !row_major_C) -> ()
  }
  %t_end_matmul = call @rtclock() : () -> f64
  %tmatmul = subf %t_end_matmul, %t_start_matmul: f64
  call @print_perf(%iters, %tmatmul) : (index, f64) -> ()

  // CHECK: {{^0$}}
  %C_ref = alloc() : !row_major_C
  linalg.fill(%C_ref, %v0) : !row_major_C, !elem_type_c
  linalg.matmul_i8_i8_i32 ins(%A, %B : !row_major_A, !row_major_B)
    outs(%C_ref: !row_major_C)
  %res = memref_cast %C : !row_major_C to memref<*xi32>
  %exp = memref_cast %C_ref : !row_major_C to memref<*xi32>
  %errors = call @verifyMemRefI32(%res, %exp) : (memref<*xi32>, memref<*xi32>) -> i64
  vector.print %errors : i64
  dealloc %C_ref : !row_major_C

  dealloc %A : !row_major_A
  dealloc %B : !row_major_B
  dealloc %C : !row_major_C

  return
}

func private @rtclock() -> f64
func private @verifyMemRefI32(memref<*xi32>, memref<*xi32>) -> i64 attributes { llvm.emit_c_interface }

// TODO: init with random, run and check output.
// func private @fill_random_f32(memref<*xf32>)
