; RUN: mlir-translate -import-llvm %s | FileCheck %s
; CHECK: module attributes {
; CHECK-SAME: llvm.target_triple = "aarch64-none-linux-android21"
target triple = "aarch64-none-linux-android21"

