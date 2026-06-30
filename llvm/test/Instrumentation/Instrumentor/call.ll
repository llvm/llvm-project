; RUN: opt -passes=instrumentor -instrumentor-read-config-files=%S/call-config.json -S < %s | FileCheck %s

declare i32 @foo(i32, ptr)
declare void @bar()

define i32 @caller(ptr %p, ptr %fp, i32 %x) {
; CHECK-LABEL: define i32 @caller(
; CHECK: call void @__instrumentor_pre_call(ptr @foo, ptr @{{.*}}, i32 2, ptr %{{.*}}, i64 4, i32 {{[0-9]+}}, i32 -1, i32 1)
; CHECK: %res = call i32 @foo(i32 %x, ptr %p)
; CHECK: call void @__instrumentor_post_call(ptr @foo, ptr @{{.*}}, i32 2, ptr %{{.*}}, i64 %{{.*}}, i64 4, i32 {{[0-9]+}}, i32 -1, i32 -1)
; CHECK: call void @__instrumentor_pre_call(ptr %fp, ptr @{{.*}}, i32 1, ptr %{{.*}}, i64 4, i32 {{[0-9]+}}, i32 -1, i32 2)
; CHECK: %ind = call i32 %fp(i32 %x)
; CHECK: call void @__instrumentor_post_call(ptr %fp, ptr @{{.*}}, i32 1, ptr %{{.*}}, i64 %{{.*}}, i64 4, i32 {{[0-9]+}}, i32 -1, i32 -2)
; CHECK: call void @__instrumentor_pre_call(ptr @bar, ptr @{{.*}}, i32 0, ptr null, i64 0, i32 {{[0-9]+}}, i32 -1, i32 3)
; CHECK: call void @bar()
; CHECK: call void @__instrumentor_post_call(ptr @bar, ptr @{{.*}}, i32 0, ptr null, i64 0, i64 0, i32 {{[0-9]+}}, i32 -1, i32 -3)
; CHECK: ret i32
  %res = call i32 @foo(i32 %x, ptr %p)
  %ind = call i32 %fp(i32 %x)
  call void @bar()
  %sum = add i32 %res, %ind
  ret i32 %sum
}

; CHECK: declare void @__instrumentor_pre_call(ptr, ptr, i32, ptr, i64, i32, i32, i32)
; CHECK: declare void @__instrumentor_post_call(ptr, ptr, i32, ptr, i64, i64, i32, i32, i32)
