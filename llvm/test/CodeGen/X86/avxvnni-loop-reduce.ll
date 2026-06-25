; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avxvnni | FileCheck %s --check-prefix=AVXVNNI
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512vnni | FileCheck %s --check-prefix=AVX512VNNI

define i32 @u8s8_loop_reduce(ptr readonly %x, ptr readonly %w, i32 %n) {
; AVXVNNI-LABEL: u8s8_loop_reduce:
; AVXVNNI:       # %bb.0: # %entry
; AVXVNNI:         {vex} vpdpbusd
; AVXVNNI:         {vex} vpdpbusd
; AVXVNNI-NOT:     vpdpwssd
; AVXVNNI:       # %bb.3: # %reduce
;
; AVX512VNNI-LABEL: u8s8_loop_reduce:
; AVX512VNNI:       # %bb.0: # %entry
; AVX512VNNI:         vpdpbusd
; AVX512VNNI-NOT:     vpdpwssd
; AVX512VNNI:       # %bb.3: # %reduce
entry:
  %has.work = icmp sgt i32 %n, 0
  br i1 %has.work, label %loop.preheader, label %exit.zero

loop.preheader:
  br label %loop

loop:
  %acc = phi <64 x i32> [ zeroinitializer, %loop.preheader ], [ %acc.next, %loop ]
  %iv = phi i32 [ 0, %loop.preheader ], [ %iv.next, %loop ]
  %iv64 = sext i32 %iv to i64
  %x.ptr = getelementptr i8, ptr %x, i64 %iv64
  %x.v = load <64 x i8>, ptr %x.ptr, align 1
  %w.ptr = getelementptr i8, ptr %w, i64 %iv64
  %w.v = load <64 x i8>, ptr %w.ptr, align 1
  %x.i32 = zext <64 x i8> %x.v to <64 x i32>
  %w.i32 = sext <64 x i8> %w.v to <64 x i32>
  %prod = mul nsw <64 x i32> %w.i32, %x.i32
  %acc.next = add <64 x i32> %prod, %acc
  %iv.next = add i32 %iv, 64
  %more = icmp slt i32 %iv.next, %n
  br i1 %more, label %loop, label %reduce

reduce:
  %sum = call i32 @llvm.vector.reduce.add.v64i32(<64 x i32> %acc.next)
  ret i32 %sum

exit.zero:
  ret i32 0
}

define i32 @u8s8_loop_reduce_extra_vector_use(ptr readonly %x, ptr readonly %w,
                                              ptr writeonly %out.vec, i32 %n) {
; AVXVNNI-LABEL: u8s8_loop_reduce_extra_vector_use:
; AVXVNNI:       # %bb.0: # %entry
; AVXVNNI-NOT:     vpdpbusd
; AVXVNNI:         vpdpwssd
;
; AVX512VNNI-LABEL: u8s8_loop_reduce_extra_vector_use:
; AVX512VNNI:       # %bb.0: # %entry
; AVX512VNNI-NOT:     vpdpbusd
; AVX512VNNI:         vpmulld
entry:
  %has.work = icmp sgt i32 %n, 0
  br i1 %has.work, label %loop.preheader, label %exit.zero

loop.preheader:
  br label %loop

loop:
  %acc = phi <64 x i32> [ zeroinitializer, %loop.preheader ], [ %acc.next, %loop ]
  %iv = phi i32 [ 0, %loop.preheader ], [ %iv.next, %loop ]
  %iv64 = sext i32 %iv to i64
  %x.ptr = getelementptr i8, ptr %x, i64 %iv64
  %x.v = load <64 x i8>, ptr %x.ptr, align 1
  %w.ptr = getelementptr i8, ptr %w, i64 %iv64
  %w.v = load <64 x i8>, ptr %w.ptr, align 1
  %x.i32 = zext <64 x i8> %x.v to <64 x i32>
  %w.i32 = sext <64 x i8> %w.v to <64 x i32>
  %prod = mul nsw <64 x i32> %w.i32, %x.i32
  %acc.next = add <64 x i32> %prod, %acc
  %iv.next = add i32 %iv, 64
  %more = icmp slt i32 %iv.next, %n
  br i1 %more, label %loop, label %reduce

reduce:
  store <64 x i32> %acc.next, ptr %out.vec, align 4
  %sum = call i32 @llvm.vector.reduce.add.v64i32(<64 x i32> %acc.next)
  ret i32 %sum

exit.zero:
  ret i32 0
}

declare i32 @llvm.vector.reduce.add.v64i32(<64 x i32>)
