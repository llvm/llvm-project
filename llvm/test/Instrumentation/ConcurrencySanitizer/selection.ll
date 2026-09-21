; RUN: opt < %s -passes='function(csan)' -S | FileCheck %s --check-prefix=DEFAULT

@constant = constant i32 0
@mutable = global i32 0

declare void @capture(ptr)

define void @uncaptured_alloca() sanitize_concurrency {
entry:
  %p = alloca i32, align 4
  store i32 1, ptr %p, align 4
  ret void
}
; DEFAULT-LABEL: @uncaptured_alloca(
; DEFAULT-NOT: __csan_write
; DEFAULT: ret void

define void @captured_alloca() sanitize_concurrency {
entry:
  %p = alloca i32, align 4
  call void @capture(ptr %p)
  store i32 1, ptr %p, align 4
  ret void
}
; DEFAULT-LABEL: @captured_alloca(
; DEFAULT: {{call|invoke}} void @capture(ptr %p)
; DEFAULT: call void @__csan_write4(ptr %p, i32 0)

define void @read_before_write(ptr %p) sanitize_concurrency {
entry:
  %v = load i32, ptr %p, align 4
  store i32 %v, ptr %p, align 4
  ret void
}
; DEFAULT-LABEL: @read_before_write(
; DEFAULT-NOT: call void @__csan_read4(ptr %p, i32 0)
; DEFAULT: call void @__csan_write4(ptr %p, i32 0)

define void @call_between_accesses(ptr %p) sanitize_concurrency {
entry:
  %v = load i32, ptr %p, align 4
  call void @capture(ptr %p)
  store i32 %v, ptr %p, align 4
  ret void
}
; DEFAULT-LABEL: @call_between_accesses(
; DEFAULT: call void @__csan_read4(ptr %p, i32 0)
; DEFAULT: {{call|invoke}} void @capture(ptr %p)
; DEFAULT: call void @__csan_write4(ptr %p, i32 0)

define i32 @constant_global() sanitize_concurrency {
entry:
  %v = load i32, ptr @constant, align 4
  ret i32 %v
}
; DEFAULT-LABEL: @constant_global(
; DEFAULT-NOT: __csan_read
; DEFAULT: ret i32

define i32 @mutable_global() sanitize_concurrency {
entry:
  %v = load i32, ptr @mutable, align 4
  ret i32 %v
}
; DEFAULT-LABEL: @mutable_global(
; DEFAULT: call void @__csan_read4(ptr @mutable, i32 0)

define void @volatile_read_before_write(ptr %p) sanitize_concurrency {
entry:
  %v = load volatile i32, ptr %p, align 4
  store volatile i32 %v, ptr %p, align 4
  ret void
}
; DEFAULT-LABEL: @volatile_read_before_write(
; DEFAULT-NOT: call void @__csan_read4(ptr %p, i32 0)
; DEFAULT: call void @__csan_write4(ptr %p, i32 0)

define void @no_sanitize_metadata(ptr %p) sanitize_concurrency {
entry:
  %v = load i32, ptr %p, align 4, !nosanitize !0
  store i32 %v, ptr %p, align 4, !nosanitize !0
  ret void
}
; DEFAULT-LABEL: @no_sanitize_metadata(
; DEFAULT-NOT: __csan_read
; DEFAULT-NOT: __csan_write
; DEFAULT: ret void

define void @unusual_size(ptr %p) sanitize_concurrency {
entry:
  %v = load i24, ptr %p, align 4
  store i24 %v, ptr %p, align 4
  ret void
}
; DEFAULT-LABEL: @unusual_size(
; DEFAULT-NOT: call void @__csan_read_range(ptr %p, i64 3, i32 0)
; DEFAULT: call void @__csan_write_range(ptr %p, i64 3, i32 0)
; DEFAULT: ret void

define void @scalable(ptr %p) sanitize_concurrency {
entry:
  %v = load <vscale x 4 x i32>, ptr %p, align 16
  store <vscale x 4 x i32> %v, ptr %p, align 16
  ret void
}
; DEFAULT-LABEL: @scalable(
; DEFAULT-NOT: call void @__csan_read_range(ptr %p, i64 %{{.*}}, i32 0)
; DEFAULT: call void @__csan_write_range(ptr %p, i64 %{{.*}}, i32 0)
; DEFAULT: ret void

define void @swifterror(ptr swifterror %p) sanitize_concurrency {
entry:
  %v = load ptr, ptr %p
  store ptr null, ptr %p
  ret void
}
; DEFAULT-LABEL: @swifterror(
; DEFAULT-NOT: __csan_read
; DEFAULT-NOT: __csan_write
; DEFAULT: ret void

define i32 @naked() naked sanitize_concurrency {
entry:
  %v = load i32, ptr @mutable
  ret i32 %v
}
; DEFAULT-LABEL: @naked(
; DEFAULT-NEXT: entry:
; DEFAULT-NEXT: %v = load i32, ptr @mutable, align 4

!0 = !{}
