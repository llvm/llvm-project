; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.lasx.xvstelm.b(<32 x i8>, ptr, i32, i32)

define void @lasx_xvstelm_b_lo(<32 x i8> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.b: argument out of range
entry:
  call void @llvm.loongarch.lasx.xvstelm.b(<32 x i8> %va, ptr %p, i32 -129, i32 1)
  ret void
}

define void @lasx_xvstelm_b_hi(<32 x i8> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.b: argument out of range
entry:
  call void @llvm.loongarch.lasx.xvstelm.b(<32 x i8> %va, ptr %p, i32 128, i32 1)
  ret void
}

define void @lasx_xvstelm_b_idx_lo(<32 x i8> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.b: argument out of range
entry:
  call void @llvm.loongarch.lasx.xvstelm.b(<32 x i8> %va, ptr %p, i32 1, i32 -1)
  ret void
}

define void @lasx_xvstelm_b_idx_hi(<32 x i8> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.b: argument out of range
entry:
  call void @llvm.loongarch.lasx.xvstelm.b(<32 x i8> %va, ptr %p, i32 1, i32 32)
  ret void
}

declare void @llvm.loongarch.lasx.xvstelm.h(<16 x i16>, ptr, i32, i32)

define void @lasx_xvstelm_h_lo(<16 x i16> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.h: argument out of range or not a multiple of 2.
entry:
  call void @llvm.loongarch.lasx.xvstelm.h(<16 x i16> %va, ptr %p, i32 -258, i32 1)
  ret void
}

define void @lasx_xvstelm_h_hi(<16 x i16> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.h: argument out of range or not a multiple of 2.
entry:
  call void @llvm.loongarch.lasx.xvstelm.h(<16 x i16> %va, ptr %p, i32 256, i32 1)
  ret void
}

define void @lasx_xvstelm_h_idx_lo(<16 x i16> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.h: argument out of range or not a multiple of 2.
entry:
  call void @llvm.loongarch.lasx.xvstelm.h(<16 x i16> %va, ptr %p, i32 2, i32 -1)
  ret void
}

define void @lasx_xvstelm_h_idx_hi(<16 x i16> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.h: argument out of range or not a multiple of 2.
entry:
  call void @llvm.loongarch.lasx.xvstelm.h(<16 x i16> %va, ptr %p, i32 2, i32 16)
  ret void
}

declare void @llvm.loongarch.lasx.xvstelm.w(<8 x i32>, ptr, i32, i32)

define void @lasx_xvstelm_w_lo(<8 x i32> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.w: argument out of range or not a multiple of 4.
entry:
  call void @llvm.loongarch.lasx.xvstelm.w(<8 x i32> %va, ptr %p, i32 -516, i32 1)
  ret void
}

define void @lasx_xvstelm_w_hi(<8 x i32> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.w: argument out of range or not a multiple of 4.
entry:
  call void @llvm.loongarch.lasx.xvstelm.w(<8 x i32> %va, ptr %p, i32 512, i32 1)
  ret void
}

define void @lasx_xvstelm_w_idx_lo(<8 x i32> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.w: argument out of range or not a multiple of 4.
entry:
  call void @llvm.loongarch.lasx.xvstelm.w(<8 x i32> %va, ptr %p, i32 4, i32 -1)
  ret void
}

define void @lasx_xvstelm_w_idx_hi(<8 x i32> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.w: argument out of range or not a multiple of 4.
entry:
  call void @llvm.loongarch.lasx.xvstelm.w(<8 x i32> %va, ptr %p, i32 4, i32 8)
  ret void
}

declare void @llvm.loongarch.lasx.xvstelm.d(<4 x i64>, ptr, i32, i32)

define void @lasx_xvstelm_d_lo(<4 x i64> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.d: argument out of range or not a multiple of 8.
entry:
  call void @llvm.loongarch.lasx.xvstelm.d(<4 x i64> %va, ptr %p, i32 -1032, i32 1)
  ret void
}

define void @lasx_xvstelm_d_hi(<4 x i64> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.d: argument out of range or not a multiple of 8.
entry:
  call void @llvm.loongarch.lasx.xvstelm.d(<4 x i64> %va, ptr %p, i32 1024, i32 1)
  ret void
}

define void @lasx_xvstelm_d_idx_lo(<4 x i64> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.d: argument out of range or not a multiple of 8.
entry:
  call void @llvm.loongarch.lasx.xvstelm.d(<4 x i64> %va, ptr %p, i32 8, i32 -1)
  ret void
}

define void @lasx_xvstelm_d_idx_hi(<4 x i64> %va, ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvstelm.d: argument out of range or not a multiple of 8.
entry:
  call void @llvm.loongarch.lasx.xvstelm.d(<4 x i64> %va, ptr %p, i32 8, i32 4)
  ret void
}
