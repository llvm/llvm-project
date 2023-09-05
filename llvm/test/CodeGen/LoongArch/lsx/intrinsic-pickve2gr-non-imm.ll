; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare i32 @llvm.loongarch.lsx.vpickve2gr.b(<16 x i8>, i32)

define i32 @lsx_vpickve2gr_b(<16 x i8> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.b(<16 x i8> %va, i32 %b)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.h(<8 x i16>, i32)

define i32 @lsx_vpickve2gr_h(<8 x i16> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.h(<8 x i16> %va, i32 %b)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.w(<4 x i32>, i32)

define i32 @lsx_vpickve2gr_w(<4 x i32> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.w(<4 x i32> %va, i32 %b)
  ret i32 %res
}

declare i64 @llvm.loongarch.lsx.vpickve2gr.d(<2 x i64>, i32)

define i64 @lsx_vpickve2gr_d(<2 x i64> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i64 @llvm.loongarch.lsx.vpickve2gr.d(<2 x i64> %va, i32 %b)
  ret i64 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.bu(<16 x i8>, i32)

define i32 @lsx_vpickve2gr_bu(<16 x i8> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.bu(<16 x i8> %va, i32 %b)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.hu(<8 x i16>, i32)

define i32 @lsx_vpickve2gr_hu(<8 x i16> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.hu(<8 x i16> %va, i32 %b)
  ret i32 %res
}

declare i32 @llvm.loongarch.lsx.vpickve2gr.wu(<4 x i32>, i32)

define i32 @lsx_vpickve2gr_wu(<4 x i32> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.lsx.vpickve2gr.wu(<4 x i32> %va, i32 %b)
  ret i32 %res
}

declare i64 @llvm.loongarch.lsx.vpickve2gr.du(<2 x i64>, i32)

define i64 @lsx_vpickve2gr_du(<2 x i64> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i64 @llvm.loongarch.lsx.vpickve2gr.du(<2 x i64> %va, i32 %b)
  ret i64 %res
}
