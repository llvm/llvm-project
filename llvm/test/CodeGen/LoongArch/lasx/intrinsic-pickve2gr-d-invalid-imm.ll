; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare i64 @llvm.loongarch.lasx.xvpickve2gr.d(<4 x i64>, i32)

define i64 @lasx_xvpickve2gr_d_lo(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve2gr.d: argument out of range
entry:
  %res = call i64 @llvm.loongarch.lasx.xvpickve2gr.d(<4 x i64> %va, i32 -1)
  ret i64 %res
}

define i64 @lasx_xvpickve2gr_d_hi(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve2gr.d: argument out of range
entry:
  %res = call i64 @llvm.loongarch.lasx.xvpickve2gr.d(<4 x i64> %va, i32 4)
  ret i64 %res
}

declare i64 @llvm.loongarch.lasx.xvpickve2gr.du(<4 x i64>, i32)

define i64 @lasx_xvpickve2gr_du_lo(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve2gr.du: argument out of range
entry:
  %res = call i64 @llvm.loongarch.lasx.xvpickve2gr.du(<4 x i64> %va, i32 -1)
  ret i64 %res
}

define i64 @lasx_xvpickve2gr_du_hi(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve2gr.du: argument out of range
entry:
  %res = call i64 @llvm.loongarch.lasx.xvpickve2gr.du(<4 x i64> %va, i32 4)
  ret i64 %res
}
