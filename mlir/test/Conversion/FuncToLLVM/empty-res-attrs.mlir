// RUN: mlir-opt %s -gpu-kernel-outlining --convert-scf-to-cf --convert-gpu-to-nvvm --nvvm-attach-target=chip=sm_75 --gpu-module-to-binary -gpu-to-llvm --convert-func-to-llvm | FileCheck %s

// Test that functions with empty res_attrs don't crash during conversion
// CHECK-LABEL: llvm.func @main()
"builtin.module"() ({
  "func.func"() ({
    "gpu.barrier"() : () -> ()
    "func.return"() : () -> ()
  }) {sym_name = "main", function_type = () -> (), arg_attrs = [], res_attrs = []} : () -> ()
}) {} : () -> ()
