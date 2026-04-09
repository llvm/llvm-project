// RUN: mlir-opt %s -split-input-file --verify-roundtrip | FileCheck %s

// CHECK: llvm.mlir.ifunc external @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver
llvm.mlir.ifunc @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver
llvm.func @resolver() -> !llvm.ptr {
  %0 = llvm.mlir.constant(333 : i64) : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// -----

// CHECK: llvm.mlir.ifunc linkonce_odr hidden @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver {dso_local}
llvm.mlir.ifunc linkonce_odr hidden @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver {dso_local}
llvm.func @resolver() -> !llvm.ptr {
  %0 = llvm.mlir.constant(333 : i64) : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// -----

// CHECK: llvm.mlir.ifunc private @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver {dso_local}
llvm.mlir.ifunc private @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver {dso_local}
llvm.func @resolver() -> !llvm.ptr {
  %0 = llvm.mlir.constant(333 : i64) : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// -----

// CHECK: llvm.mlir.ifunc weak @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver
llvm.mlir.ifunc weak @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver
llvm.func @resolver() -> !llvm.ptr {
  %0 = llvm.mlir.constant(333 : i64) : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

