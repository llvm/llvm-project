; RUN: opt < %s -passes=tsan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i16 @test_unaligned_read2(i16* %a) sanitize_thread {
entry:
  %tmp1 = load i16, i16* %a, align 1
  ret i16 %tmp1
}

; CHECK-LABEL: define i16 @test_unaligned_read2(i16* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i16* %a to i8*
; CHECK-NEXT:   call void @__tsan_unaligned_read2(i8* %1)
; CHECK-NEXT:   %tmp1 = load i16, i16* %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i16

define i32 @test_unaligned_read4(i32* %a) sanitize_thread {
entry:
  %tmp1 = load i32, i32* %a, align 2
  ret i32 %tmp1
}

; CHECK-LABEL: define i32 @test_unaligned_read4(i32* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i32* %a to i8*
; CHECK-NEXT:   call void @__tsan_unaligned_read4(i8* %1)
; CHECK-NEXT:   %tmp1 = load i32, i32* %a, align 2
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i32

define i64 @test_unaligned_read8(i64* %a) sanitize_thread {
entry:
  %tmp1 = load i64, i64* %a, align 4
  ret i64 %tmp1
}

; CHECK-LABEL: define i64 @test_unaligned_read8(i64* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i64* %a to i8*
; CHECK-NEXT:   call void @__tsan_unaligned_read8(i8* %1)
; CHECK-NEXT:   %tmp1 = load i64, i64* %a, align 4
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i64

define i128 @test_unaligned_read16(i128* %a) sanitize_thread {
entry:
  %tmp1 = load i128, i128* %a, align 1
  ret i128 %tmp1
}

; CHECK-LABEL: define i128 @test_unaligned_read16(i128* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i128* %a to i8*
; CHECK-NEXT:   call void @__tsan_unaligned_read16(i8* %1)
; CHECK-NEXT:   %tmp1 = load i128, i128* %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i128

define i128 @test_aligned_read16(i128* %a) sanitize_thread {
entry:
  %tmp1 = load i128, i128* %a, align 8
  ret i128 %tmp1
}

; CHECK-LABEL: define i128 @test_aligned_read16(i128* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i128* %a to i8*
; CHECK-NEXT:   call void @__tsan_read16(i8* %1)
; CHECK-NEXT:   %tmp1 = load i128, i128* %a, align 8
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i128

define void @test_unaligned_write2(i16* %a) sanitize_thread {
entry:
  store i16 1, i16* %a, align 1
  ret void
}

; CHECK-LABEL: define void @test_unaligned_write2(i16* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i16* %a to i8*
; CHECK-NEXT:   call void @__tsan_unaligned_write2(i8* %1)
; CHECK-NEXT:   store i16 1, i16* %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void

define void @test_unaligned_write4(i32* %a) sanitize_thread {
entry:
  store i32 1, i32* %a, align 1
  ret void
}

; CHECK-LABEL: define void @test_unaligned_write4(i32* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i32* %a to i8*
; CHECK-NEXT:   call void @__tsan_unaligned_write4(i8* %1)
; CHECK-NEXT:   store i32 1, i32* %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void

define void @test_unaligned_write8(i64* %a) sanitize_thread {
entry:
  store i64 1, i64* %a, align 1
  ret void
}

; CHECK-LABEL: define void @test_unaligned_write8(i64* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i64* %a to i8*
; CHECK-NEXT:   call void @__tsan_unaligned_write8(i8* %1)
; CHECK-NEXT:   store i64 1, i64* %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void

define void @test_unaligned_write16(i128* %a) sanitize_thread {
entry:
  store i128 1, i128* %a, align 1
  ret void
}

; CHECK-LABEL: define void @test_unaligned_write16(i128* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i128* %a to i8*
; CHECK-NEXT:   call void @__tsan_unaligned_write16(i8* %1)
; CHECK-NEXT:   store i128 1, i128* %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void

define void @test_aligned_write16(i128* %a) sanitize_thread {
entry:
  store i128 1, i128* %a, align 8
  ret void
}

; CHECK-LABEL: define void @test_aligned_write16(i128* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i128* %a to i8*
; CHECK-NEXT:   call void @__tsan_write16(i8* %1)
; CHECK-NEXT:   store i128 1, i128* %a, align 8
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void
