; RUN: opt < %s -passes='function(csan)' -S | FileCheck %s

target datalayout = "e-p:64:64"

define void @three_byte_access(ptr %src, ptr %dst) sanitize_concurrency {
entry:
  %value = load i24, ptr %src, align 1
  store i24 %value, ptr %dst, align 1
  ret void
}
; CHECK-LABEL: @three_byte_access(
; CHECK: call void @__csan_read_range(ptr %src, i64 3, i32 0)
; CHECK: call void @__csan_write_range(ptr %dst, i64 3, i32 0)

define void @large_access(ptr %src, ptr %dst) sanitize_concurrency {
entry:
  %value = load i256, ptr %src, align 32
  store i256 %value, ptr %dst, align 32
  ret void
}
; CHECK-LABEL: @large_access(
; CHECK: call void @__csan_read_range(ptr %src, i64 32, i32 0)
; CHECK: call void @__csan_write_range(ptr %dst, i64 32, i32 0)

define void @scalable_access(ptr %src, ptr %dst) sanitize_concurrency {
entry:
  %value = load <vscale x 8 x i32>, ptr %src, align 32
  store <vscale x 8 x i32> %value, ptr %dst, align 32
  ret void
}
; CHECK-LABEL: @scalable_access(
; CHECK: call void @__csan_read_range(ptr %src, i64 %{{.*}}, i32 0)
; CHECK: call void @__csan_write_range(ptr %dst, i64 %{{.*}}, i32 0)
