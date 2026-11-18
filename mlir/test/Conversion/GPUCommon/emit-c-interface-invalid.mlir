// RUN: mlir-opt %s --gpu-to-llvm -split-input-file | FileCheck %s

// The C interface wrapper signature requires every argument type to be LLVM
// compatible, which is stricter than the plain signature conversion: this
// pass converts the function signature but cannot build a wrapper for it.
// The function must be left unconverted instead of creating a wrapper with a
// null function type.

// CHECK-LABEL: func.func @ciface_unsupported_arg
// CHECK-NOT:     _mlir_ciface_
func.func @ciface_unsupported_arg(%arg0: !async.token)
    attributes {llvm.emit_c_interface} {
  return
}

// -----

// Same for an external function, which takes the other wrapper code path.

// CHECK-LABEL: func.func private @ciface_unsupported_arg_decl
// CHECK-NOT:     _mlir_ciface_
func.func private @ciface_unsupported_arg_decl(%arg0: !async.token)
    attributes {llvm.emit_c_interface}

// -----

// A function whose arguments do convert must still get its wrapper.

// CHECK-LABEL: llvm.func @ciface_supported_arg
// CHECK:       llvm.func @_mlir_ciface_ciface_supported_arg
func.func @ciface_supported_arg(%arg0: memref<4xf32>)
    attributes {llvm.emit_c_interface} {
  return
}
