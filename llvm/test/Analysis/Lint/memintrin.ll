; RUN: opt -passes=lint -disable-output < %s 2>&1 | FileCheck %s

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) nounwind argmemonly

define void @f_memcpy() {
entry:
  %dst = alloca [1000 x i8], align 2
  %src = alloca [1000 x i8], align 4
; CHECK: Undefined behavior: Memory reference address is misaligned
; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 %dst, ptr align 4 %src, i32 200, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %dst, ptr align 4 %src, i32 200, i1 false)
; CHECK: Undefined behavior: Memory reference address is misaligned
; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 2 %dst, ptr align 8 %src, i32 200, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 2 %dst, ptr align 8 %src, i32 200, i1 false)
; CHECK-NOT: @llvm.memcpy.p0.p0.i32
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %dst, ptr align 4 %src, i32 200, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 2 %dst, ptr align 2 %src, i32 200, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 2 %dst, ptr align 4 %src, i32 200, i1 false)

  ret void
}

declare void @llvm.memmove.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) nounwind argmemonly

define void @f_memmove() {
entry:
  %dst = alloca [1000 x i8], align 2
  %src = alloca [1000 x i8], align 4
; CHECK: Undefined behavior: Memory reference address is misaligned
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 200, i1 false)
  call void @llvm.memmove.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 200, i1 false)
; CHECK: Undefined behavior: Memory reference address is misaligned
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i32(ptr align 2 %dst, ptr align 8 %src, i32 200, i1 false)
  call void @llvm.memmove.p0.p0.i32(ptr align 2 %dst, ptr align 8 %src, i32 200, i1 false)
; CHECK-NOT: @llvm.memmove.p0.p0.i32
  call void @llvm.memmove.p0.p0.i32(ptr align 1 %dst, ptr align 4 %src, i32 200, i1 false)
  call void @llvm.memmove.p0.p0.i32(ptr align 2 %dst, ptr align 2 %src, i32 200, i1 false)
  call void @llvm.memmove.p0.p0.i32(ptr align 2 %dst, ptr align 4 %src, i32 200, i1 false)

  ret void
}

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1) nounwind argmemonly

define void @f_memset() {
entry:
  %dst = alloca [1000 x i8], align 2
; CHECK: Undefined behavior: Memory reference address is misaligned
; CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 4 %dst, i8 55, i32 200, i1 false)
  call void @llvm.memset.p0.i32(ptr align 4 %dst, i8 55, i32 200, i1 false)
; CHECK-NOT: @llvm.memset.p0.i32
  call void @llvm.memset.p0.i32(ptr align 1 %dst, i8 55, i32 200, i1 false)
  call void @llvm.memset.p0.i32(ptr align 2 %dst, i8 55, i32 200, i1 false)

  ret void
}
