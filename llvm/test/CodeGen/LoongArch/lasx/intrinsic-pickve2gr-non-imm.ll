; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare i32 @llvm.loongarch.lasx.xvpickve2gr.w(<8 x i32>, i32)

define i32 @lasx_xvpickve2gr_w(<8 x i32> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.lasx.xvpickve2gr.w(<8 x i32> %va, i32 %b)
  ret i32 %res
}

declare i64 @llvm.loongarch.lasx.xvpickve2gr.d(<4 x i64>, i32)

define i64 @lasx_xvpickve2gr_d(<4 x i64> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i64 @llvm.loongarch.lasx.xvpickve2gr.d(<4 x i64> %va, i32 %b)
  ret i64 %res
}

declare i32 @llvm.loongarch.lasx.xvpickve2gr.wu(<8 x i32>, i32)

define i32 @lasx_xvpickve2gr_wu(<8 x i32> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.lasx.xvpickve2gr.wu(<8 x i32> %va, i32 %b)
  ret i32 %res
}

declare i64 @llvm.loongarch.lasx.xvpickve2gr.du(<4 x i64>, i32)

define i64 @lasx_xvpickve2gr_du(<4 x i64> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i64 @llvm.loongarch.lasx.xvpickve2gr.du(<4 x i64> %va, i32 %b)
  ret i64 %res
}
