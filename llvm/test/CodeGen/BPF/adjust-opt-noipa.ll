; RUN: opt -O2 -mtriple=bpf-pc-linux -S < %s | FileCheck %s
;
; Every global function definition gets noipa, so that no IPO pass changes
; what its callers pass to it or assume about it. Static functions and
; declarations are left alone.

; CHECK: define dso_local i32 @global_func(ptr noundef %ctx, i32 noundef %x) {{.*}}#[[NOIPA:[0-9]+]]
define dso_local i32 @global_func(ptr noundef %ctx, i32 noundef %x) noinline {
  ret i32 0
}

; CHECK: define weak i32 @weak_func(ptr noundef %ctx) {{.*}}#[[NOIPA]]
define weak i32 @weak_func(ptr noundef %ctx) noinline {
  ret i32 0
}

; CHECK: define internal fastcc i32 @static_func() {{.*}}#[[NONOIPA:[0-9]+]]
define internal i32 @static_func(ptr noundef %ctx) noinline {
  %r = call i32 @extern_func(ptr null)
  ret i32 %r
}

; The unused arguments of @global_func and @weak_func are kept, but not the
; one of @static_func.
; CHECK-LABEL: define dso_local i32 @caller(
; CHECK:         call i32 @global_func(ptr noundef %ctx, i32 noundef 1)
; CHECK:         call i32 @weak_func(ptr noundef %ctx)
; CHECK:         call fastcc i32 @static_func()
define dso_local i32 @caller(ptr noundef %ctx) {
  %a = call i32 @global_func(ptr noundef %ctx, i32 noundef 1)
  %b = call i32 @weak_func(ptr noundef %ctx)
  %c = call i32 @static_func(ptr noundef %ctx)
  %ab = add i32 %a, %b
  %r = add i32 %ab, %c
  ret i32 %r
}

; CHECK: declare i32 @extern_func(ptr) local_unnamed_addr{{$}}
declare i32 @extern_func(ptr)

; CHECK: attributes #[[NOIPA]] = { {{.*}}noipa
; CHECK-NOT: attributes #[[NONOIPA]] = { {{.*}}noipa
