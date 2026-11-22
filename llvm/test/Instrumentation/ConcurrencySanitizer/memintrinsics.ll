; RUN: opt < %s -passes='function(csan)' -csan-instrument-func-entry-exit=0 -S -mtriple=amdgpu-amd-amdhsa | FileCheck %s

define void @instrument_source(ptr addrspace(5) %dst, ptr addrspace(1) %src, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.memcpy.p5.p1.i32(ptr addrspace(5) %dst, ptr addrspace(1) %src, i32 %n, i1 false)
  ret void
}
; CHECK-LABEL: @instrument_source(
; CHECK: %[[LEN:.*]] = zext i32 %n to i64
; CHECK: %[[SRC:.*]] = addrspacecast ptr addrspace(1) %src to ptr
; CHECK: call void @__csan_read_range(ptr %[[SRC]], i64 %[[LEN]], i32 0)
; CHECK-NOT: __csan_write_range
; CHECK: call void @llvm.memcpy

define void @instrument_dest(ptr addrspace(1) %dst, ptr addrspace(5) %src, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.memcpy.p1.p5.i32(ptr addrspace(1) %dst, ptr addrspace(5) %src, i32 %n, i1 false)
  ret void
}
; CHECK-LABEL: @instrument_dest(
; CHECK: %[[LEN:.*]] = zext i32 %n to i64
; CHECK-NOT: __csan_read_range
; CHECK: %[[DST:.*]] = addrspacecast ptr addrspace(1) %dst to ptr
; CHECK: call void @__csan_write_range(ptr %[[DST]], i64 %[[LEN]], i32 0)
; CHECK: call void @llvm.memcpy

define void @private_copy(ptr addrspace(5) %dst, ptr addrspace(5) %src, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.memcpy.p5.p5.i32(ptr addrspace(5) %dst, ptr addrspace(5) %src, i32 %n, i1 false)
  ret void
}
; CHECK-LABEL: @private_copy(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.memcpy.p5.p5.i32
; CHECK-NEXT: ret void

define void @instrument_memset(ptr addrspace(1) %dst, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.memset.p1.i32(ptr addrspace(1) %dst, i8 0, i32 %n, i1 false)
  ret void
}
; CHECK-LABEL: @instrument_memset(
; CHECK: %[[LEN:.*]] = zext i32 %n to i64
; CHECK: %[[DST:.*]] = addrspacecast ptr addrspace(1) %dst to ptr
; CHECK: call void @__csan_write_range(ptr %[[DST]], i64 %[[LEN]], i32 0)
; CHECK: call void @llvm.memset

define void @atomic_memcpy(ptr addrspace(1) %dst, ptr addrspace(1) %src, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.memcpy.element.unordered.atomic.p1.p1.i32(ptr addrspace(1) align 4 %dst, ptr addrspace(1) align 4 %src, i32 %n, i32 4)
  ret void
}
; CHECK-LABEL: @atomic_memcpy(
; CHECK: %[[LEN:.*]] = zext i32 %n to i64
; CHECK: %[[SRC:.*]] = addrspacecast ptr addrspace(1) %src to ptr
; CHECK: call void @__csan_read_range(ptr %[[SRC]], i64 %[[LEN]], i32 1)
; CHECK: %[[DST:.*]] = addrspacecast ptr addrspace(1) %dst to ptr
; CHECK: call void @__csan_write_range(ptr %[[DST]], i64 %[[LEN]], i32 1)
; CHECK: call void @llvm.memcpy.element.unordered.atomic

define void @atomic_memmove(ptr addrspace(1) %dst, ptr addrspace(5) %src, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.memmove.element.unordered.atomic.p1.p5.i32(ptr addrspace(1) align 4 %dst, ptr addrspace(5) align 4 %src, i32 %n, i32 4)
  ret void
}
; CHECK-LABEL: @atomic_memmove(
; CHECK-NOT: __csan_read_range
; CHECK: call void @__csan_write_range(ptr %{{.*}}, i64 %{{.*}}, i32 1)
; CHECK: call void @llvm.memmove.element.unordered.atomic

define void @atomic_memset(ptr addrspace(1) %dst, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.memset.element.unordered.atomic.p1.i32(ptr addrspace(1) align 4 %dst, i8 0, i32 %n, i32 4)
  ret void
}
; CHECK-LABEL: @atomic_memset(
; CHECK: call void @__csan_write_range(ptr %{{.*}}, i64 %{{.*}}, i32 1)
; CHECK: call void @llvm.memset.element.unordered.atomic

define void @memset_pattern(ptr addrspace(1) %dst, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.experimental.memset.pattern.p1.i64.i32(ptr addrspace(1) %dst, i64 7, i32 %n, i1 false)
  ret void
}
; CHECK-LABEL: @memset_pattern(
; CHECK: %[[COUNT:.*]] = zext i32 %n to i64
; CHECK: %[[LEN:.*]] = mul i64 %[[COUNT]], 8
; CHECK: %[[DST:.*]] = addrspacecast ptr addrspace(1) %dst to ptr
; CHECK: call void @__csan_write_range(ptr %[[DST]], i64 %[[LEN]], i32 0)
; CHECK: call void @llvm.experimental.memset.pattern

define void @memset_pattern_private(ptr addrspace(5) %dst, i32 %n) sanitize_concurrency {
entry:
  call void @llvm.experimental.memset.pattern.p5.i64.i32(ptr addrspace(5) %dst, i64 7, i32 %n, i1 false)
  ret void
}
; CHECK-LABEL: @memset_pattern_private(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.experimental.memset.pattern
; CHECK-NEXT: ret void
