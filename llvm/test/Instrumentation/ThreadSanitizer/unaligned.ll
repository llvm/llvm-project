; RUN: opt < %s -passes=tsan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i16 @test_unaligned_read2(ptr %a) sanitize_thread {
entry:
  %tmp1 = load i16, ptr %a, align 1
  ret i16 %tmp1
}

; CHECK-LABEL: define i16 @test_unaligned_read2(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_unaligned_read2(ptr %a)
; CHECK-NEXT:   %tmp1 = load i16, ptr %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i16

define i32 @test_unaligned_read4(ptr %a) sanitize_thread {
entry:
  %tmp1 = load i32, ptr %a, align 2
  ret i32 %tmp1
}

; CHECK-LABEL: define i32 @test_unaligned_read4(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_unaligned_read4(ptr %a)
; CHECK-NEXT:   %tmp1 = load i32, ptr %a, align 2
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i32

define i64 @test_unaligned_read8(ptr %a) sanitize_thread {
entry:
  %tmp1 = load i64, ptr %a, align 4
  ret i64 %tmp1
}

; CHECK-LABEL: define i64 @test_unaligned_read8(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_unaligned_read8(ptr %a)
; CHECK-NEXT:   %tmp1 = load i64, ptr %a, align 4
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i64

define i128 @test_unaligned_read16(ptr %a) sanitize_thread {
entry:
  %tmp1 = load i128, ptr %a, align 1
  ret i128 %tmp1
}

; CHECK-LABEL: define i128 @test_unaligned_read16(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_unaligned_read16(ptr %a)
; CHECK-NEXT:   %tmp1 = load i128, ptr %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i128

define i128 @test_aligned_read16(ptr %a) sanitize_thread {
entry:
  %tmp1 = load i128, ptr %a, align 8
  ret i128 %tmp1
}

; CHECK-LABEL: define i128 @test_aligned_read16(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_read16(ptr %a)
; CHECK-NEXT:   %tmp1 = load i128, ptr %a, align 8
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i128

define void @test_unaligned_write2(ptr %a) sanitize_thread {
entry:
  store i16 1, ptr %a, align 1
  ret void
}

; CHECK-LABEL: define void @test_unaligned_write2(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_unaligned_write2(ptr %a)
; CHECK-NEXT:   store i16 1, ptr %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void

define void @test_unaligned_write4(ptr %a) sanitize_thread {
entry:
  store i32 1, ptr %a, align 1
  ret void
}

; CHECK-LABEL: define void @test_unaligned_write4(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_unaligned_write4(ptr %a)
; CHECK-NEXT:   store i32 1, ptr %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void

define void @test_unaligned_write8(ptr %a) sanitize_thread {
entry:
  store i64 1, ptr %a, align 1
  ret void
}

; CHECK-LABEL: define void @test_unaligned_write8(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_unaligned_write8(ptr %a)
; CHECK-NEXT:   store i64 1, ptr %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void

define void @test_unaligned_write16(ptr %a) sanitize_thread {
entry:
  store i128 1, ptr %a, align 1
  ret void
}

; CHECK-LABEL: define void @test_unaligned_write16(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_unaligned_write16(ptr %a)
; CHECK-NEXT:   store i128 1, ptr %a, align 1
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void

define void @test_aligned_write16(ptr %a) sanitize_thread {
entry:
  store i128 1, ptr %a, align 8
  ret void
}

; CHECK-LABEL: define void @test_aligned_write16(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_write16(ptr %a)
; CHECK-NEXT:   store i128 1, ptr %a, align 8
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret void
