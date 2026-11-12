; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare i64 @llvm.loongarch.lsx.vpickve2gr.d(<2 x i64>, i32)

define i64 @lsx_vpickve2gr_d_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.d: argument out of range
entry:
  %res = call i64 @llvm.loongarch.lsx.vpickve2gr.d(<2 x i64> %va, i32 -1)
  ret i64 %res
}

define i64 @lsx_vpickve2gr_d_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.d: argument out of range
entry:
  %res = call i64 @llvm.loongarch.lsx.vpickve2gr.d(<2 x i64> %va, i32 2)
  ret i64 %res
}

declare i64 @llvm.loongarch.lsx.vpickve2gr.du(<2 x i64>, i32)

define i64 @lsx_vpickve2gr_du_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.du: argument out of range
entry:
  %res = call i64 @llvm.loongarch.lsx.vpickve2gr.du(<2 x i64> %va, i32 -1)
  ret i64 %res
}

define i64 @lsx_vpickve2gr_du_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.du: argument out of range
entry:
  %res = call i64 @llvm.loongarch.lsx.vpickve2gr.du(<2 x i64> %va, i32 2)
  ret i64 %res
}
