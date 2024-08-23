; RUN: llc < %s -verify-machineinstrs

; Check that llc doesn't crash.

target triple = "nvptx64-nvidia-cuda"

define void @__builtin_splat_i8(i32 %0) {
.lr.ph:
  %1 = trunc i32 %0 to i8
  %broadcast.splatinsert = insertelement <4 x i8> poison, i8 %1, i64 0
  %broadcast.splat = shufflevector <4 x i8> %broadcast.splatinsert, <4 x i8> poison, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:
  store <4 x i8> %broadcast.splat, ptr addrspace(1) poison, align 1
  br label %vector.body
}
