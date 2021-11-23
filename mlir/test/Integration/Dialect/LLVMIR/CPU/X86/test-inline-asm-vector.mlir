// RUN: mlir-opt %s -convert-vector-to-llvm |  \
// RUN: mlir-cpu-runner -e entry_point_with_all_constants -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext

module {
  llvm.func @function_to_run(%a: vector<8xf32>, %b: vector<8xf32>)  {
    // CHECK: ( 8, 10, 12, 14, 16, 18, 20, 22 )
    %r0 = llvm.inline_asm asm_dialect = intel
        "vaddps $0, $1, $2", "=x,x,x" %a, %b:
      (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    vector.print %r0: vector<8xf32>

    // vblendps implemented with inline_asm.
    // CHECK: ( 0, 1, 10, 11, 4, 5, 14, 15 )
    %r1 = llvm.inline_asm asm_dialect = intel
        "vblendps $0, $1, $2, 0xCC", "=x,x,x" %a, %b:
      (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    vector.print %r1: vector<8xf32>

    // vblendps 0xCC via vector.shuffle (emulates clang intrinsics impl)
    // CHECK: ( 0, 1, 10, 11, 4, 5, 14, 15 )
    %r2 = vector.shuffle %a, %b[0, 1, 10, 11, 4, 5, 14, 15]
      : vector<8xf32>, vector<8xf32>
    vector.print %r2: vector<8xf32>

    // vblendps 0x33 implemented with inline_asm.
    // CHECK: ( 8, 9, 2, 3, 12, 13, 6, 7 )
    %r3 = llvm.inline_asm asm_dialect = intel
        "vblendps $0, $1, $2, 0x33", "=x,x,x" %a, %b:
      (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    vector.print %r3: vector<8xf32>

    // vblendps 0x33 via vector.shuffle (emulates clang intrinsics impl)
    // CHECK: ( 8, 9, 2, 3, 12, 13, 6, 7 )
    %r4 = vector.shuffle %a, %b[8, 9, 2, 3, 12, 13, 6, 7]
      : vector<8xf32>, vector<8xf32>
    vector.print %r4: vector<8xf32>

    llvm.return
  }

  // Solely exists to prevent inlining and get the expected assembly.
  llvm.func @entry_point(%a: vector<8xf32>, %b: vector<8xf32>)  {
    llvm.call @function_to_run(%a, %b) : (vector<8xf32>, vector<8xf32>) -> ()
    llvm.return
  }

  llvm.func @entry_point_with_all_constants()  {
    %a = llvm.mlir.constant(dense<[0.0, 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0]>
      : vector<8xf32>) : vector<8xf32>
    %b = llvm.mlir.constant(dense<[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]>
      : vector<8xf32>) : vector<8xf32>
    llvm.call @function_to_run(%a, %b) : (vector<8xf32>, vector<8xf32>) -> ()
    llvm.return
  }
}
