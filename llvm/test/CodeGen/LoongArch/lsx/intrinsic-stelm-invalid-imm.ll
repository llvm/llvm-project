; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.lsx.vstelm.b(<16 x i8>, ptr, i32, i32)

define void @lsx_vstelm_b_lo(<16 x i8> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.b: argument out of range
entry:
  call void @llvm.loongarch.lsx.vstelm.b(<16 x i8> %va, ptr %p, i32 -129, i32 15)
  ret void
}

define void @lsx_vstelm_b_hi(<16 x i8> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.b: argument out of range
entry:
  call void @llvm.loongarch.lsx.vstelm.b(<16 x i8> %va, ptr %p, i32 128, i32 15)
  ret void
}

define void @lsx_vstelm_b_idx_lo(<16 x i8> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.b: argument out of range
entry:
  call void @llvm.loongarch.lsx.vstelm.b(<16 x i8> %va, ptr %p, i32 1, i32 -1)
  ret void
}

define void @lsx_vstelm_b_idx_hi(<16 x i8> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.b: argument out of range
entry:
  call void @llvm.loongarch.lsx.vstelm.b(<16 x i8> %va, ptr %p, i32 1, i32 16)
  ret void
}

declare void @llvm.loongarch.lsx.vstelm.h(<8 x i16>, ptr, i32, i32)

define void @lsx_vstelm_h_lo(<8 x i16> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.h: argument out of range or not a multiple of 2.
entry:
  call void @llvm.loongarch.lsx.vstelm.h(<8 x i16> %va, ptr %p, i32 -258, i32 7)
  ret void
}

define void @lsx_vstelm_h_hi(<8 x i16> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.h: argument out of range or not a multiple of 2.
entry:
  call void @llvm.loongarch.lsx.vstelm.h(<8 x i16> %va, ptr %p, i32 256, i32 7)
  ret void
}

define void @lsx_vstelm_h_idx_lo(<8 x i16> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.h: argument out of range or not a multiple of 2.
entry:
  call void @llvm.loongarch.lsx.vstelm.h(<8 x i16> %va, ptr %p, i32 2, i32 -1)
  ret void
}

define void @lsx_vstelm_h_idx_hi(<8 x i16> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.h: argument out of range or not a multiple of 2.
entry:
  call void @llvm.loongarch.lsx.vstelm.h(<8 x i16> %va, ptr %p, i32 2, i32 8)
  ret void
}

declare void @llvm.loongarch.lsx.vstelm.w(<4 x i32>, ptr, i32, i32)

define void @lsx_vstelm_w_lo(<4 x i32> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.w: argument out of range or not a multiple of 4.
entry:
  call void @llvm.loongarch.lsx.vstelm.w(<4 x i32> %va, ptr %p, i32 -516, i32 3)
  ret void
}

define void @lsx_vstelm_w_hi(<4 x i32> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.w: argument out of range or not a multiple of 4.
entry:
  call void @llvm.loongarch.lsx.vstelm.w(<4 x i32> %va, ptr %p, i32 512, i32 3)
  ret void
}

define void @lsx_vstelm_w_idx_lo(<4 x i32> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.w: argument out of range or not a multiple of 4.
entry:
  call void @llvm.loongarch.lsx.vstelm.w(<4 x i32> %va, ptr %p, i32 4, i32 -1)
  ret void
}

define void @lsx_vstelm_w_idx_hi(<4 x i32> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.w: argument out of range or not a multiple of 4.
entry:
  call void @llvm.loongarch.lsx.vstelm.w(<4 x i32> %va, ptr %p, i32 4, i32 4)
  ret void
}

declare void @llvm.loongarch.lsx.vstelm.d(<2 x i64>, ptr, i32, i32)

define void @lsx_vstelm_d_lo(<2 x i64> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.d: argument out of range or not a multiple of 8.
entry:
  call void @llvm.loongarch.lsx.vstelm.d(<2 x i64> %va, ptr %p, i32 -1032, i32 1)
  ret void
}

define void @lsx_vstelm_d_hi(<2 x i64> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.d: argument out of range or not a multiple of 8.
entry:
  call void @llvm.loongarch.lsx.vstelm.d(<2 x i64> %va, ptr %p, i32 1024, i32 1)
  ret void
}

define void @lsx_vstelm_d_idx_lo(<2 x i64> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.d: argument out of range or not a multiple of 8.
entry:
  call void @llvm.loongarch.lsx.vstelm.d(<2 x i64> %va, ptr %p, i32 8, i32 -1)
  ret void
}

define void @lsx_vstelm_d_idx_hi(<2 x i64> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lsx.vstelm.d: argument out of range or not a multiple of 8.
entry:
  call void @llvm.loongarch.lsx.vstelm.d(<2 x i64> %va, ptr %p, i32 8, i32 2)
  ret void
}
