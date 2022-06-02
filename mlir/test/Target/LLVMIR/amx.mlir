// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @target(ptr %0)
// CHECK: %[[c:.*]] = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 16)
// CHECK: call void @llvm.x86.tilestored64.internal(i16 16, i16 16, ptr %0, i64 32, x86_amx %[[c]]
llvm.func @target(%ptr: !llvm.ptr<i8>) {
  %c = llvm.mlir.constant(16 : i16) : i16
  %s = llvm.mlir.constant(32 : i64) : i64
  %0 = "amx.tilezero"(%c, %c) : (i16, i16) -> !llvm.array<16 x vector<16xbf16>>
  "amx.tilestored64"(%c, %c, %ptr, %s, %0) : (i16, i16, !llvm.ptr<i8>, i64, !llvm.array<16 x vector<16xbf16>>) -> ()
  llvm.return
}

