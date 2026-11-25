; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target triple = "riscv64"

; CHECK: new-vl result of llvm.riscv.vleff/vleff.mask must fit in XLEN
define <vscale x 2 x i32> @standard_vleff(<vscale x 2 x i32> %p, ptr %a, i128 %vl) {
    %s = call { <vscale x 2 x i32>, i128 } @llvm.riscv.vleff.nxv2i32(
        <vscale x 2 x i32> %p, ptr %a, i128 %vl)
    %v = extractvalue { <vscale x 2 x i32>, i128 } %s, 0
    ret <vscale x 2 x i32> %v
}

; CHECK: new-vl result of llvm.riscv.vleff/vleff.mask must fit in XLEN
define <vscale x 2 x i32> @mask_vleff(<vscale x 2 x i32> %p, ptr %a, <vscale x 2 x i1> %m, i128 %vl) {
    %s = call { <vscale x 2 x i32>, i128 } @llvm.riscv.vleff.mask.nxv2i32(
    <vscale x 2 x i32> %p, ptr %a, <vscale x 2 x i1> %m, i128 %vl, i128 1)
    %v = extractvalue { <vscale x 2 x i32>, i128 } %s, 0
    ret <vscale x 2 x i32> %v
}
