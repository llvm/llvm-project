; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.lsx.vstelm.b(<16 x i8>, i8*, i32, i32)

define void @lsx_vstelm_b(<16 x i8> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vstelm.b(<16 x i8> %va, i8* %p, i32 %b, i32 1)
  ret void
}

define void @lsx_vstelm_b_idx(<16 x i8> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vstelm.b(<16 x i8> %va, i8* %p, i32 1, i32 %b)
  ret void
}

declare void @llvm.loongarch.lsx.vstelm.h(<8 x i16>, i8*, i32, i32)

define void @lsx_vstelm_h(<8 x i16> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vstelm.h(<8 x i16> %va, i8* %p, i32 %b, i32 1)
  ret void
}

define void @lsx_vstelm_h_idx(<8 x i16> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vstelm.h(<8 x i16> %va, i8* %p, i32 2, i32 %b)
  ret void
}

declare void @llvm.loongarch.lsx.vstelm.w(<4 x i32>, i8*, i32, i32)

define void @lsx_vstelm_w(<4 x i32> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vstelm.w(<4 x i32> %va, i8* %p, i32 %b, i32 1)
  ret void
}

define void @lsx_vstelm_w_idx(<4 x i32> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vstelm.w(<4 x i32> %va, i8* %p, i32 4, i32 %b)
  ret void
}

declare void @llvm.loongarch.lsx.vstelm.d(<2 x i64>, i8*, i32, i32)

define void @lsx_vstelm_d(<2 x i64> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vstelm.d(<2 x i64> %va, i8* %p, i32 %b, i32 1)
  ret void
}

define void @lsx_vstelm_d_idx(<2 x i64> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vstelm.d(<2 x i64> %va, i8* %p, i32 8, i32 %b)
  ret void
}
