; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.lasx.xvstelm.b(<32 x i8>, ptr, i32, i32)

define void @lasx_xvstelm_b(<32 x i8> %va, ptr %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvstelm.b(<32 x i8> %va, ptr %p, i32 %b, i32 1)
  ret void
}

define void @lasx_xvstelm_b_idx(<32 x i8> %va, ptr %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvstelm.b(<32 x i8> %va, ptr %p, i32 1, i32 %b)
  ret void
}

declare void @llvm.loongarch.lasx.xvstelm.h(<16 x i16>, ptr, i32, i32)

define void @lasx_xvstelm_h(<16 x i16> %va, ptr %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvstelm.h(<16 x i16> %va, ptr %p, i32 %b, i32 1)
  ret void
}

define void @lasx_xvstelm_h_idx(<16 x i16> %va, ptr %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvstelm.h(<16 x i16> %va, ptr %p, i32 2, i32 %b)
  ret void
}

declare void @llvm.loongarch.lasx.xvstelm.w(<8 x i32>, ptr, i32, i32)

define void @lasx_xvstelm_w(<8 x i32> %va, ptr %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvstelm.w(<8 x i32> %va, ptr %p, i32 %b, i32 1)
  ret void
}

define void @lasx_xvstelm_w_idx(<8 x i32> %va, ptr %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvstelm.w(<8 x i32> %va, ptr %p, i32 4, i32 %b)
  ret void
}

declare void @llvm.loongarch.lasx.xvstelm.d(<4 x i64>, ptr, i32, i32)

define void @lasx_xvstelm_d(<4 x i64> %va, ptr %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvstelm.d(<4 x i64> %va, ptr %p, i32 %b, i32 1)
  ret void
}

define void @lasx_xvstelm_d_idx(<4 x i64> %va, ptr %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvstelm.d(<4 x i64> %va, ptr %p, i32 8, i32 %b)
  ret void
}
