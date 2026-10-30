; RUN: not opt -mtriple=riscv32 -passes=verify -disable-output < %s 2>&1 | FileCheck %s

; CHECK: llvm.riscv.vsetvli/vsetvlimax result must be i32 on riscv32
; CHECK-NEXT: %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %avl, i64 0, i64 0)
define i64 @vsetvli_i64(i64 %avl) {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %avl, i64 0, i64 0)
  ret i64 %vl
}

; CHECK: llvm.riscv.vsetvli/vsetvlimax result must be i32 on riscv32
; CHECK-NEXT: %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 0, i64 0)
define i64 @vsetvlimax_i64() {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 0, i64 0)
  ret i64 %vl
}
