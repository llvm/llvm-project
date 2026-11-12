; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

%"bucket<string, double, '\\b'>::Iterator" = type { ptr, i64, i64 }

; CHECK-LABEL: llvm.func @g
define void @g() {
  %item.i = alloca %"bucket<string, double, '\\b'>::Iterator", align 8
  ; CHECK: llvm.alloca %0 x !llvm.struct<"bucket<string, double, '\\b'>::Iterator", (ptr, i64, i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  ret void
}
