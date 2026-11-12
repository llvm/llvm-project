; RUN: sed 's/iXLEN/i32/g' %s | not opt -passes=verify -mtriple riscv32 -S 2>&1 | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: sed 's/iXLEN/i64/g' %s | not opt -passes=verify -mtriple riscv64 -S 2>&1 | FileCheck %s --check-prefixes=CHECK,RV64

; CHECK: llvm.riscv.vsetvli/vsetvlimax result must be XLEN
define i16 @narrow_result() {
  %vl = call i16 @llvm.riscv.vsetvlimax.i16(i16 0, i16 3)
  ret i16 %vl
}

; RV64: llvm.riscv.vsetvli/vsetvlimax result must be XLEN
; RV64-NEXT: %vl = call i32 @llvm.riscv.vsetvli.i32(i32 %avl, i32 0, i32 0)
define i32 @vsetvli_i32(i32 %avl) {
  %vl = call i32 @llvm.riscv.vsetvli.i32(i32 %avl, i32 0, i32 0)
  ret i32 %vl
}

; RV32: llvm.riscv.vsetvli/vsetvlimax result must be XLEN
; RV32-NEXT: %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %avl, i64 0, i64 0)
define i64 @vsetvli_i64(i64 %avl) {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %avl, i64 0, i64 0)
  ret i64 %vl
}

; CHECK: llvm.riscv.vsetvli/vsetvlimax VSEW must be 0-3
define iXLEN @bad_vsew() {
  %vl = call iXLEN @llvm.riscv.vsetvlimax.iXLEN(iXLEN 5, iXLEN 0)
  ret iXLEN %vl
}

; CHECK: llvm.riscv.vsetvli/vsetvlimax VLMUL is reserved
define iXLEN @reserved_vlmul() {
  %vl = call iXLEN @llvm.riscv.vsetvlimax.iXLEN(iXLEN 0, iXLEN 4)
  ret iXLEN %vl
}

; CHECK: llvm.riscv.vsetvli/vsetvlimax VSEW must be 0-3
define iXLEN @vsetvli_bad_vsew(iXLEN %avl) {
  %vl = call iXLEN @llvm.riscv.vsetvli.iXLEN(iXLEN %avl, iXLEN 7, iXLEN 0)
  ret iXLEN %vl
}
