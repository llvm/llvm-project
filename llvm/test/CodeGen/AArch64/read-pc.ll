; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s

define i64 @read_pc() {
  ; CHECK: adr x0, #0
  %pc = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %pc
}

declare i64 @llvm.read_register.i64(metadata) nounwind

!0 = !{!"pc"}
