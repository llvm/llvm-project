; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lsx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare i32 @llvm.loongarch.lsx.vpickve2gr.b(<16 x i8>, i32)

define i32 @lsx_vpickve2gr_b_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.b: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.b(<16 x i8> %va, i32 -1)
  ret i32 %res
}

define i32 @lsx_vpickve2gr_b_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.b: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.b(<16 x i8> %va, i32 16)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.h(<8 x i16>, i32)

define i32 @lsx_vpickve2gr_h_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.h: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.h(<8 x i16> %va, i32 -1)
  ret i32 %res
}

define i32 @lsx_vpickve2gr_h_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.h: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.h(<8 x i16> %va, i32 8)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.w(<4 x i32>, i32)

define i32 @lsx_vpickve2gr_w_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.w: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.w(<4 x i32> %va, i32 -1)
  ret i32 %res
}

define i32 @lsx_vpickve2gr_w_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.w: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.w(<4 x i32> %va, i32 4)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.bu(<16 x i8>, i32)

define i32 @lsx_vpickve2gr_bu_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.bu: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.bu(<16 x i8> %va, i32 -1)
  ret i32 %res
}

define i32 @lsx_vpickve2gr_bu_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.bu: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.bu(<16 x i8> %va, i32 16)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.hu(<8 x i16>, i32)

define i32 @lsx_vpickve2gr_hu_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.hu: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.hu(<8 x i16> %va, i32 -1)
  ret i32 %res
}

define i32 @lsx_vpickve2gr_hu_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.hu: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.hu(<8 x i16> %va, i32 8)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.wu(<4 x i32>, i32)

define i32 @lsx_vpickve2gr_wu_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.wu: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.wu(<4 x i32> %va, i32 -1)
  ret i32 %res
}

define i32 @lsx_vpickve2gr_wu_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vpickve2gr.wu: argument out of range
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.wu(<4 x i32> %va, i32 4)
  ret i32 %res
}
