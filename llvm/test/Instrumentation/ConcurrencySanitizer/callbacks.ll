; RUN: opt < %s -passes='function(csan)' -csan-distinguish-volatile -S | FileCheck %s

define void @access_sizes(ptr %p1, ptr %p2, ptr %p4, ptr %p8, ptr %p16) sanitize_concurrency {
entry:
  %v1 = load i8, ptr %p1, align 1
  %v2 = load i16, ptr %p2, align 2
  %v4 = load i32, ptr %p4, align 4
  %v8 = load i64, ptr %p8, align 8
  %v16 = load i128, ptr %p16, align 16
  store i8 %v1, ptr %p1, align 1
  store i16 %v2, ptr %p2, align 2
  store i32 %v4, ptr %p4, align 4
  store i64 %v8, ptr %p8, align 8
  store i128 %v16, ptr %p16, align 16
  ret void
}
; CHECK-LABEL: @access_sizes(
; CHECK-DAG: call void @__csan_write1(ptr %p1, i32 0)
; CHECK-DAG: call void @__csan_write2(ptr %p2, i32 0)
; CHECK-DAG: call void @__csan_write4(ptr %p4, i32 0)
; CHECK-DAG: call void @__csan_write8(ptr %p8, i32 0)
; CHECK-DAG: call void @__csan_write16(ptr %p16, i32 0)

define i128 @read_sizes(ptr %p1, ptr %p8, ptr %p16) sanitize_concurrency {
entry:
  %v1 = load i8, ptr %p1, align 1
  %v8 = load i64, ptr %p8, align 8
  %v16 = load i128, ptr %p16, align 16
  %v1.ext = zext i8 %v1 to i128
  %v8.ext = zext i64 %v8 to i128
  %sum1 = add i128 %v1.ext, %v8.ext
  %sum2 = add i128 %sum1, %v16
  ret i128 %sum2
}
; CHECK-LABEL: @read_sizes(
; CHECK: call void @__csan_read1(ptr %p1, i32 0)
; CHECK: call void @__csan_read8(ptr %p8, i32 0)
; CHECK: call void @__csan_read16(ptr %p16, i32 0)

define i32 @volatile_read(ptr %p) sanitize_concurrency {
entry:
  %v = load volatile i32, ptr %p, align 4
  ret i32 %v
}
; CHECK-LABEL: @volatile_read(
; CHECK: call void @__csan_volatile_read4(ptr %p, i32 0)

define void @unaligned_volatile_write(ptr %p) sanitize_concurrency {
entry:
  store volatile i64 1, ptr %p, align 1
  ret void
}
; CHECK-LABEL: @unaligned_volatile_write(
; CHECK: call void @__csan_unaligned_volatile_write8(ptr %p, i32 0)

define i32 @unaligned_atomic_rmw(ptr %p) sanitize_concurrency {
entry:
  %v = atomicrmw add ptr %p, i32 1 seq_cst, align 1
  ret i32 %v
}
; CHECK-LABEL: @unaligned_atomic_rmw(
; CHECK: call void @__csan_unaligned_read_write4(ptr %p, i32 3)
; CHECK-NEXT: %v = atomicrmw add ptr %p, i32 1 seq_cst, align 1

define float @atomic_float(ptr %p) sanitize_concurrency {
entry:
  %v = load atomic float, ptr %p monotonic, align 4
  ret float %v
}
; CHECK-LABEL: @atomic_float(
; CHECK: call void @__csan_read4(ptr %p, i32 1)
; CHECK-NEXT: %v = load atomic float, ptr %p monotonic, align 4
