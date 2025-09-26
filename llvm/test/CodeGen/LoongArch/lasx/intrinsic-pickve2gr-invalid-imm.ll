; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
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
