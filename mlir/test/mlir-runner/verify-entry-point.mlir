// RUN: not mlir-runner %s -e entry_point_void -entry-point-result=void 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-POINT-VOID
// RUN: not mlir-runner %s -e entry_inputs_void -entry-point-result=void 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-INPUTS-VOID
// RUN: not mlir-runner %s -e entry_result_void -entry-point-result=void 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-RESULT-VOID
// RUN: not mlir-runner %s -e entry_point_i32 -entry-point-result=i32 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-POINT-I32
// RUN: not mlir-runner %s -e entry_inputs_i32 -entry-point-result=i32 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-INPUTS-I32
// RUN: not mlir-runner %s -e entry_result_i32 -entry-point-result=i32 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-RESULT-I32
// RUN: not mlir-runner %s -e entry_result_i64 -entry-point-result=i64 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-RESULT-I64
// RUN: not mlir-runner %s -e entry_result_f32 -entry-point-result=f32 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-RESULT-F32

// CHECK-ENTRY-POINT-VOID: Error: entry point not found
llvm.func @entry_point_void() -> ()

// CHECK-ENTRY-INPUTS-VOID: Error: JIT can't invoke a main function expecting arguments
llvm.func @entry_inputs_void(%arg0: i32) {
  llvm.return
}

// CHECK-ENTRY-RESULT-VOID: Error: expected void function
llvm.func @entry_result_void() -> (i32) {
  %0 = llvm.mlir.constant(0 : index) : i32
  llvm.return %0 : i32
}

// CHECK-ENTRY-POINT-I32: Error: entry point not found
llvm.func @entry_point_i32() -> (i32)

// CHECK-ENTRY-INPUTS-I32: Error: JIT can't invoke a main function expecting arguments
llvm.func @entry_inputs_i32(%arg0: i32) {
  llvm.return
}

// CHECK-ENTRY-RESULT-I32: Error: only single i32 function result supported
llvm.func @entry_result_i32() -> (i64) {
  %0 = llvm.mlir.constant(0 : index) : i64
  llvm.return %0 : i64
}

// CHECK-ENTRY-RESULT-I64: Error: only single i64 function result supported
llvm.func @entry_result_i64() -> (i32) {
  %0 = llvm.mlir.constant(0 : index) : i32
  llvm.return %0 : i32
}

// CHECK-ENTRY-RESULT-F32: Error: only single f32 function result supported
llvm.func @entry_result_f32() -> (i32) {
  %0 = llvm.mlir.constant(0 : index) : i32
  llvm.return %0 : i32
}
