// RUN: mlir-cpu-runner %s %if target={{s390x-.*}} %{ -argext-abi-check=false %} \
// RUN:   | FileCheck %s
// RUN: mlir-cpu-runner %s -e foo %if target={{s390x-.*}} %{ -argext-abi-check=false %} \
// RUN:   | FileCheck -check-prefix=NOMAIN %s
// RUN: mlir-cpu-runner %s --entry-point-result=i32 -e int32_main %if target={{s390x-.*}} \
// RUN:   %{ -argext-abi-check=false %} | FileCheck -check-prefix=INT32MAIN %s
// RUN: mlir-cpu-runner %s --entry-point-result=i64 -e int64_main %if target={{s390x-.*}} \
// RUN:   %{ -argext-abi-check=false %} | FileCheck -check-prefix=INT64MAIN %s
// RUN: mlir-cpu-runner %s -O3 %if target={{s390x-.*}} %{ -argext-abi-check=false %} \
// RUN:   | FileCheck %s

// RUN: cp %s %t
// RUN: mlir-cpu-runner %t -dump-object-file %if target={{s390x-.*}} \
// RUN:   %{ -argext-abi-check=false %} | FileCheck %t
// RUN: ls %t.o
// RUN: rm %t.o

// RUN: mlir-cpu-runner %s -dump-object-file -object-filename=%T/test.o \
// RUN:   %if target={{s390x-.*}} %{ -argext-abi-check=false %} | FileCheck %s
// RUN: ls %T/test.o
// RUN: rm %T/test.o

// Declarations of C library functions.
llvm.func @logbf(f32) -> f32
llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @free(!llvm.ptr)

// Check that a simple function with a nested call works.
llvm.func @main() -> f32 {
  %0 = llvm.mlir.constant(-4.200000e+02 : f32) : f32
  %1 = llvm.call @logbf(%0) : (f32) -> f32
  llvm.return %1 : f32
}
// CHECK: 8.000000e+00

// Helper typed functions wrapping calls to "malloc" and "free".
llvm.func @allocation() -> !llvm.ptr {
  %0 = llvm.mlir.constant(4 : index) : i64
  %1 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
  llvm.return %1 : !llvm.ptr
}
llvm.func @deallocation(%arg0: !llvm.ptr) {
  llvm.call @free(%arg0) : (!llvm.ptr) -> ()
  llvm.return
}

// Check that allocation and deallocation works, and that a custom entry point
// works.
llvm.func @foo() -> f32 {
  %0 = llvm.call @allocation() : () -> !llvm.ptr
  %1 = llvm.mlir.constant(0 : index) : i64
  %2 = llvm.mlir.constant(1.234000e+03 : f32) : f32
  %3 = llvm.getelementptr %0[%1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %2, %3 : f32, !llvm.ptr
  %4 = llvm.getelementptr %0[%1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %5 = llvm.load %4 : !llvm.ptr -> f32
  llvm.call @deallocation(%0) : (!llvm.ptr) -> ()
  llvm.return %5 : f32
}
// NOMAIN: 1.234000e+03

// Check that i32 return type works
llvm.func @int32_main() -> i32 {
  %0 = llvm.mlir.constant(42 : i32) : i32
  llvm.return %0 : i32
}
// INT32MAIN: 42

// Check that i64 return type works
llvm.func @int64_main() -> i64 {
  %0 = llvm.mlir.constant(42 : i64) : i64
  llvm.return %0 : i64
}
// INT64MAIN: 42
