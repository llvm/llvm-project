; RUN: opt -passes=instrumentor -instrumentor-read-config-files=%S/call-name-filter-config.json -S < %s | FileCheck %s

declare i32 @foo(i32)
declare i32 @bar(i32)

define i32 @caller(ptr %fp, i32 %x) {
; CHECK-LABEL: define i32 @caller(
; CHECK: call void @__instrumentor_pre_call(ptr @foo, ptr @{{.*}}, i32 1)
; CHECK-NEXT: %foo = call i32 @foo(i32 %x)
; CHECK-NEXT: call void @__instrumentor_post_call(ptr @foo, ptr @{{.*}}, i32 -1)
; CHECK-NEXT: %bar = call i32 @bar(i32 %x)
; CHECK-NEXT: %ind = call i32 %fp(i32 %x)
; CHECK-NEXT: %sum = add i32 %foo, %bar
; CHECK-NEXT: %ret = add i32 %sum, %ind
; CHECK-NEXT: ret i32 %ret
  %foo = call i32 @foo(i32 %x)
  %bar = call i32 @bar(i32 %x)
  %ind = call i32 %fp(i32 %x)
  %sum = add i32 %foo, %bar
  %ret = add i32 %sum, %ind
  ret i32 %ret
}

; CHECK: declare void @__instrumentor_pre_call(ptr, ptr, i32)
; CHECK: declare void @__instrumentor_post_call(ptr, ptr, i32)
; CHECK-NOT: __instrumentor_pre_call(ptr @bar
; CHECK-NOT: __instrumentor_post_call(ptr @bar
; CHECK-NOT: __instrumentor_pre_call(ptr %fp
; CHECK-NOT: __instrumentor_post_call(ptr %fp
