; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s

declare i16 @llvm.riscv.vsetvlimax.i16(i16, i16)
declare i64 @llvm.riscv.vsetvlimax.i64(i64, i64)
declare i64 @llvm.riscv.vsetvli.i64(i64, i64, i64)

; CHECK: llvm.riscv.vsetvli/vsetvlimax result must be i32 or i64
define i16 @narrow_result() {
  %vl = call i16 @llvm.riscv.vsetvlimax.i16(i16 0, i16 3)
  ret i16 %vl
}

; CHECK: llvm.riscv.vsetvli/vsetvlimax VSEW must be 0-3
define i64 @bad_vsew() {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 5, i64 0)
  ret i64 %vl
}

; CHECK: llvm.riscv.vsetvli/vsetvlimax VLMUL is reserved
define i64 @reserved_vlmul() {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 0, i64 4)
  ret i64 %vl
}

; CHECK: llvm.riscv.vsetvli/vsetvlimax VSEW must be 0-3
define i64 @vsetvli_bad_vsew(i64 %avl) {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %avl, i64 7, i64 0)
  ret i64 %vl
}
