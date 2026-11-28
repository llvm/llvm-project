; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target triple = "riscv32"

; CHECK: llvm.riscv.vleff/vleff.mask can't make use of i64 for riscv32
define <vscale x 2 x i32> @standard_vleff(<vscale x 2 x i32> %p, ptr %a, i64 %vl) {
  %s = call { <vscale x 2 x i32>, i64 } @llvm.riscv.vleff.nxv2i32(
         <vscale x 2 x i32> %p, ptr %a, i64 %vl)
  %v = extractvalue { <vscale x 2 x i32>, i64 } %s, 0
  ret <vscale x 2 x i32> %v
}

; CHECK: llvm.riscv.vleff/vleff.mask can't make use of i64 for riscv32
define <vscale x 2 x i32> @mask_vleff(<vscale x 2 x i32> %p, ptr %a, <vscale x 2 x i1> %m, i64 %vl) {
 %s = call { <vscale x 2 x i32>, i64 } @llvm.riscv.vleff.mask.nxv2i32(
    <vscale x 2 x i32> %p, ptr %a, <vscale x 2 x i1> %m, i64 %vl, i64 1)
    %v = extractvalue { <vscale x 2 x i32>, i64 } %s, 0
    ret <vscale x 2 x i32> %v
}
