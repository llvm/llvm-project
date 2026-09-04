; RUN: mlir-translate -import-llvm %s | FileCheck %s
; CHECK: llvm.module_asm = ["foo", "bar"]

module asm "foo"
module asm "bar"
