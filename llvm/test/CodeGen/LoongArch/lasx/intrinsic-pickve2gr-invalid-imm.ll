; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare i32 @llvm.loongarch.lasx.xvpickve2gr.w(<8 x i32>, i32)

define i32 @lasx_xvpickve2gr_w_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve2gr.w: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lasx.xvpickve2gr.w(<8 x i32> %va, i32 -1)
  ret i32 %res
}

define i32 @lasx_xvpickve2gr_w_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve2gr.w: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lasx.xvpickve2gr.w(<8 x i32> %va, i32 8)
  ret i32 %res
}

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

declare i32 @llvm.loongarch.lasx.xvpickve2gr.wu(<8 x i32>, i32)

define i32 @lasx_xvpickve2gr_wu_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve2gr.wu: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lasx.xvpickve2gr.wu(<8 x i32> %va, i32 -1)
  ret i32 %res
}

define i32 @lasx_xvpickve2gr_wu_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve2gr.wu: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lasx.xvpickve2gr.wu(<8 x i32> %va, i32 8)
  ret i32 %res
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
