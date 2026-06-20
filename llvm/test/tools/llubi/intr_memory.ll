; RUN: llubi --verbose < %s 2>&1 | FileCheck %s

declare void @llvm.memset.p0.i16(ptr, i8, i16, i1)
declare void @llvm.memset.inline.p0.i8(ptr, i8, i8, i1)
declare void @llvm.memcpy.p0.p0.i8(ptr, ptr, i8, i1)
declare void @llvm.memcpy.inline.p0.p0.i8(ptr, ptr, i8, i1)
declare void @llvm.memmove.p0.p0.i8(ptr, ptr, i8, i1)

define void @main() {
  %src = alloca [8 x i8], align 1
  %dst = alloca [8 x i8], align 1
  call void @llvm.memset.p0.i16(ptr %src, i8 17, i16 8, i1 false)
  call void @llvm.memcpy.p0.p0.i8(ptr %dst, ptr %src, i8 8, i1 false)

  %dst2 = getelementptr i8, ptr %dst, i64 2
  call void @llvm.memset.inline.p0.i8(ptr %dst2, i8 34, i8 3, i1 false)

  %src4 = getelementptr i8, ptr %src, i64 4
  %dst5 = getelementptr i8, ptr %dst, i64 5
  call void @llvm.memcpy.inline.p0.p0.i8(ptr %dst5, ptr %src4, i8 2, i1 false)

  %v0 = load i8, ptr %dst, align 1
  %p1 = getelementptr i8, ptr %dst, i64 1
  %v1 = load i8, ptr %p1, align 1
  %p2 = getelementptr i8, ptr %dst, i64 2
  %v2 = load i8, ptr %p2, align 1
  %p4 = getelementptr i8, ptr %dst, i64 4
  %v4 = load i8, ptr %p4, align 1
  %p5 = getelementptr i8, ptr %dst, i64 5
  %v5 = load i8, ptr %p5, align 1

  %move = alloca [6 x i8], align 1
  store i8 1, ptr %move, align 1
  %m1 = getelementptr i8, ptr %move, i64 1
  store i8 2, ptr %m1, align 1
  %m2 = getelementptr i8, ptr %move, i64 2
  store i8 3, ptr %m2, align 1
  %m3 = getelementptr i8, ptr %move, i64 3
  store i8 4, ptr %m3, align 1
  %m4 = getelementptr i8, ptr %move, i64 4
  store i8 5, ptr %m4, align 1
  %m5 = getelementptr i8, ptr %move, i64 5
  store i8 6, ptr %m5, align 1
  call void @llvm.memmove.p0.p0.i8(ptr %m1, ptr %move, i8 5, i1 false)

  %mv0 = load i8, ptr %move, align 1
  %mv1 = load i8, ptr %m1, align 1
  %mv2 = load i8, ptr %m2, align 1
  %mv5 = load i8, ptr %m5, align 1

  call void @llvm.memset.p0.i16(ptr poison, i8 0, i16 0, i1 false)
  call void @llvm.memcpy.p0.p0.i8(ptr poison, ptr poison, i8 0, i1 false)

  ret void
}

; CHECK: Entering function: main
; CHECK:   call void @llvm.memset.p0.i16(ptr %src, i8 17, i16 8, i1 false)
; CHECK:   call void @llvm.memcpy.p0.p0.i8(ptr %dst, ptr %src, i8 8, i1 false)
; CHECK:   call void @llvm.memset.inline.p0.i8(ptr %dst2, i8 34, i8 3, i1 false)
; CHECK:   call void @llvm.memcpy.inline.p0.p0.i8(ptr %dst5, ptr %src4, i8 2, i1 false)
; CHECK:   %v0 = load i8, ptr %dst, align 1 => i8 17
; CHECK:   %v1 = load i8, ptr %p1, align 1 => i8 17
; CHECK:   %v2 = load i8, ptr %p2, align 1 => i8 34
; CHECK:   %v4 = load i8, ptr %p4, align 1 => i8 34
; CHECK:   %v5 = load i8, ptr %p5, align 1 => i8 17
; CHECK:   call void @llvm.memmove.p0.p0.i8(ptr %m1, ptr %move, i8 5, i1 false)
; CHECK:   %mv0 = load i8, ptr %move, align 1 => i8 1
; CHECK:   %mv1 = load i8, ptr %m1, align 1 => i8 1
; CHECK:   %mv2 = load i8, ptr %m2, align 1 => i8 2
; CHECK:   %mv5 = load i8, ptr %m5, align 1 => i8 5
; CHECK:   call void @llvm.memset.p0.i16(ptr poison, i8 0, i16 0, i1 false)
; CHECK:   call void @llvm.memcpy.p0.p0.i8(ptr poison, ptr poison, i8 0, i1 false)
; CHECK:   ret void
; CHECK: Exiting function: main
